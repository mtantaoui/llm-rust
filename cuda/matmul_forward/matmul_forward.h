void matmul_forward(int kernel_num, float *out, const float *inp,
                    const float *weight, const float *bias, int B, int T, int C,
                    int OC, const int sqrt_block_size);