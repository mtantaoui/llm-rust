#include "adamw/adamw.h"
#include "matmul_forward/matmul_forward.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef COMMON
#define COMMON

template <class T> __host__ __device__ T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
static void cuda_check(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
static void cublasCheck(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
    printf("code: %d, reason: %s\n", status, cublasGetStatusString(status));
    exit(EXIT_FAILURE);
  }
}
#define cublasCheck(status)                                                    \
  { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4
// is OK
static size_t cublaslt_workspace_size = 32 * 1024 * 1024;
static void *cublaslt_workspace = NULL;
static cublasComputeType_t cublas_compute_type;

static cublasHandle_t cublas_handle;
static cublasLtHandle_t cublaslt_handle;

static int cuda_arch_major = 0;
static int cuda_arch_minor = 0;
static int cuda_num_SMs =
    0; // for persistent threads where we want 1 threadblock per SM
static int cuda_threads_per_SM =
    0; // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// random utils

static float *make_random_float_01(size_t N) {
  float *arr = (float *)malloc(N * sizeof(float));
  for (size_t i = 0; i < N; i++) {
    arr[i] = ((float)rand() / RAND_MAX); // range 0..1
  }
  return arr;
}

static float *make_random_float(size_t N) {
  float *arr = (float *)malloc(N * sizeof(float));
  for (size_t i = 0; i < N; i++) {
    arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
  }
  return arr;
}

static int *make_random_int(size_t N, int V) {
  int *arr = (int *)malloc(N * sizeof(int));
  for (size_t i = 0; i < N; i++) {
    arr[i] = rand() % V; // range 0..V-1
  }
  return arr;
}

static float *make_zeros_float(size_t N) {
  float *arr = (float *)malloc(N * sizeof(float));
  memset(arr, 0, N * sizeof(float)); // all zero
  return arr;
}

static float *make_ones_float(size_t N) {
  float *arr = (float *)malloc(N * sizeof(float));
  for (size_t i = 0; i < N; i++) {
    arr[i] = 1.0f;
  }
  return arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template <class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType *d_ptr, float *h_ptr,
                                         size_t count) {
  // copy from host to device with data type conversion.
  TargetType *converted = (TargetType *)malloc(count * sizeof(TargetType));
  for (int i = 0; i < count; i++) {
    converted[i] = (TargetType)h_ptr[i];
  }

  cudaError_t status = cudaMemcpy(d_ptr, converted, count * sizeof(TargetType),
                                  cudaMemcpyHostToDevice);
  free(converted);

  // instead of checking the status at cudaMemcpy, we return it from here.
  // This way, we still need to use our checking macro, and get better line
  // info as to where the error happened.
  return status;
}

static void setup_main() {
  srand(0); // determinism

  // set up the device
  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceIdx);
  cuda_num_SMs = deviceProp.multiProcessorCount;
  cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
  cuda_arch_major = deviceProp.major;
  cuda_arch_minor = deviceProp.minor;

  // setup cuBLAS and cuBLASLt
  cublasCheck(cublasCreate(&cublas_handle));
  cublasCheck(cublasLtCreate(&cublaslt_handle));
  cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

  // TF32 precision is equivalent to
  // torch.set_float32_matmul_precision('high')
  int enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
  // TODO implement common CLI for all tests/benchmarks
  // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via
  // arg
  cublas_compute_type =
      enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  cublasMath_t cublas_math_mode =
      enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

template <class D, class T>
void validate_result(D *device_result, const T *cpu_reference, const char *name,
                     std::size_t num_elements, T tolerance = 1e-4) {
  D *out_gpu = (D *)malloc(num_elements * sizeof(D));
  cudaCheck(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(D),
                       cudaMemcpyDeviceToHost));
  int nfaults = 0;
  for (int i = 0; i < num_elements; i++) {
    // print the first few comparisons
    if (i < 5) {
      printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
    }
    // ensure correctness for all elements. We can set an "ignore" mask by
    // writing NaN
    if (fabs(cpu_reference[i] - (T)out_gpu[i]) > tolerance &&
        isfinite(cpu_reference[i])) {
      printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i,
             cpu_reference[i], (T)out_gpu[i]);
      nfaults++;
      if (nfaults >= 10) {
        free(out_gpu);
        exit(EXIT_FAILURE);
      }
    }
  }
  printf("\n");

  // reset the result pointer, so we can chain multiple tests and don't miss
  // trivial errors, like the kernel not writing to part of the result.
  // cudaMemset(device_result, 0, num_elements * sizeof(T));
  // AK: taking this out, ~2 hours of my life was spent finding this line

  free(out_gpu);
}

template <class Kernel, class... KernelArgs>
static float benchmark_kernel(int repeats, Kernel kernel,
                              KernelArgs &&...kernel_args) {
  cudaEvent_t start, stop;
  // prepare buffer to scrub L2 cache between benchmarks
  // just memset a large dummy array, recommended by
  // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
  // and apparently used in nvbench.
  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaDeviceProp deviceProp;
  cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
  void *flush_buffer;
  cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));
  float elapsed_time = 0.f;
  for (int i = 0; i < repeats; i++) {
    // clear L2
    cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
    // now we can start recording the timing of the kernel
    cudaCheck(cudaEventRecord(start, nullptr));
    kernel(std::forward<KernelArgs>(kernel_args)...);
    cudaCheck(cudaEventRecord(stop, nullptr));
    cudaCheck(cudaEventSynchronize(start));
    cudaCheck(cudaEventSynchronize(stop));
    float single_call;
    cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
    elapsed_time += single_call;
  }

  cudaCheck(cudaFree(flush_buffer));

  return elapsed_time / repeats;
}

#endif