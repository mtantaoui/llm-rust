#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "../common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void adamw_cpu(float *params_memory, const float *grads_memory, float *m_memory, float *v_memory, int t, long num_parameters, float learning_rate = 1e-3, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8, float weight_decay = 0.0)
{
    // adapted from: train_gpt2.c

    for (int i = 0; i < num_parameters; i++)
    {
        float param = params_memory[i];
        float grad = grads_memory[i];

        // update the first moment (momentum)
        float m = beta1 * m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        float v = beta2 * v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        // update
        m_memory[i] = m;
        v_memory[i] = v;
        params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// utility functions

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
__device__ inline float lerp(float start, float end, float weight)
{
    return fma(weight, end, fma(-weight, start, start));
}

// naive fused kernel
__global__ void adamw_kernel1(float *params_memory, const float *grads_memory, float *m_memory, float *v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_parameters)
        return; // guard
    // update the first moment (momentum)
    m_memory[i] = beta1 * m_memory[i] + (1.0f - beta1) * grads_memory[i];
    // update the second moment (RMSprop)
    v_memory[i] = beta2 * v_memory[i] + (1.0f - beta2) * grads_memory[i] * grads_memory[i];
    float m_hat = m_memory[i] / beta1_correction;
    float v_hat = v_memory[i] / beta2_correction;
    params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * params_memory[i]);
}

// Slightly more optimized AdamW kernel by:
// * loading data that is accessed more than once into registers,
// * using optimized linear interpolation for the moment updates.
__global__ void adamw_kernel2(float *params_memory, const float *grads_memory, float *m_memory, float *v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_parameters)
        return; // guard
    float grad = grads_memory[i];
    float m = m_memory[i];
    float v = v_memory[i];
    // update the first moment (momentum)
    m = lerp(grad, m, beta1);
    m_memory[i] = m;
    // update the second moment (RMSprop)
    v = lerp(grad * grad, v, beta2);
    v_memory[i] = v;
    m /= beta1_correction; // m_hat
    v /= beta2_correction; // v_hat
    params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}

// ----------------------------------------------------------------------------
// kernel launcher

// version 1: naive dispatch to naive kernel
void adamw_dispatch1(float *params_memory, const float *grads_memory, float *m_memory, float *v_memory, long num_parameters,
                     float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay)
{
    unsigned int block_size = 512;
    unsigned int num_blocks = ceil_div(num_parameters, (long)block_size);
    adamw_kernel1<<<num_blocks, block_size>>>(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                                              learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

// version 2: naive dispatch to slightly optimized kernel
void adamw_dispatch2(float *params_memory, const float *grads_memory, float *m_memory, float *v_memory, long num_parameters,
                     float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay)
{
    unsigned int block_size = 512;
    unsigned int num_blocks = ceil_div(num_parameters, (long)block_size);
    adamw_kernel2<<<num_blocks, block_size>>>(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                                              learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

void adamw_optimizer(int kernel_num,
                     float *params_memory, const float *grads_memory, float *m_memory, float *v_memory, int t, long num_parameters,
                     float learning_rate = 1e-3, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8, float weight_decay = 0.0)
{
    // calculate the m_hat and v_hat correction terms once as they are the same for every param/thread
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    switch (kernel_num)
    {
    case 1:
        adamw_dispatch1(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                        learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
        break;
    case 2:
        adamw_dispatch2(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                        learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
        break;
    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
}

void adamw(int kernel_num)
{
    printf("caca \n");
}
