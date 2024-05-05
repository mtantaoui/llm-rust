
void adamw(int kernel_num, float *params_memory, const float *grads_memory,
           float *m_memory, float *v_memory, int t, long num_parameters,
           float learning_rate, float beta1, float beta2, float eps,
           float weight_decay);