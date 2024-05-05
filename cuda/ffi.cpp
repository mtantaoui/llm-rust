#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include "adamw/adamw.h"
#include "matmul_forward/matmul_forward.h"

using namespace std;

extern "C"
{
    void matmul_forward_cuda(int kernel_num,
                             float *out,
                             const float *inp, const float *weight, const float *bias,
                             int B, int T, int C, int OC,
                             const int sqrt_block_size)
    {
        matmul_forward(kernel_num, out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
    }

    void adamw_cuda(int kernel_num)
    {
        adamw(kernel_num);
    }
}