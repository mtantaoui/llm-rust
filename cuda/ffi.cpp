#include "adamw/adamw.h"
#include "matmul_forward/matmul_forward.h"

using namespace std;

extern "C" {
void matmul_forward_cuda(int kernel_num, float *out, const float *inp,
                         const float *weight, const float *bias, int B, int T,
                         int C, int OC, const int sqrt_block_size) {
    matmul_forward(kernel_num, out, inp, weight, bias, B, T, C, OC,
                   sqrt_block_size);
}

void adamw_cuda(int kernel_num, float *params_memory, const float *grads_memory,
                float *m_memory, float *v_memory, int t, long num_parameters,
                float learning_rate = 1e-3, float beta1 = 0.9,
                float beta2 = 0.999, float eps = 1e-8,
                float weight_decay = 0.0) {

    adamw(kernel_num, params_memory, grads_memory, m_memory, v_memory, t,
          num_parameters, learning_rate, beta1, beta2, eps, weight_decay);
}
}