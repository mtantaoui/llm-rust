use std::ffi::{c_float, c_int, c_long};

use num::ToPrimitive;

use crate::utils::{make_random_float, make_random_float_01};

extern "C" {

    fn adamw_cuda(
        kernel_num: c_int,
        params_memory: *mut c_float,
        grads_memory: *mut c_float,
        m_memory: *mut c_float,
        v_memory: *mut c_float,
        t: c_int,
        num_parameters: c_long,
        learning_rate: c_float,
        beta1: c_float,
        beta2: c_float,
        eps: c_float,
        weight_decay: c_float,
    );

}

fn adamw_test() {
    let num_parameters: usize = 1048576;
    let t = 10;
    let learning_rate = 1e-3f32;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps = 1e-8f32;
    let weight_decay = 0.0f32;

    // create random data on host (to be used for the CPU reference implementation)
    let mut params_memory = make_random_float(num_parameters);
    let mut grads_memory = make_random_float(num_parameters);
    let mut m_memory = make_random_float_01(num_parameters);
    let mut v_memory = make_random_float_01(num_parameters);

    unsafe {
        adamw_cuda(
            1,
            params_memory.as_mut_ptr(),
            grads_memory.as_mut_ptr(),
            m_memory.as_mut_ptr(),
            v_memory.as_mut_ptr(),
            t,
            num_parameters.to_i64().unwrap(),
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
        );
        adamw_cuda(
            2,
            params_memory.as_mut_ptr(),
            grads_memory.as_mut_ptr(),
            m_memory.as_mut_ptr(),
            v_memory.as_mut_ptr(),
            t,
            num_parameters.to_i64().unwrap(),
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
        )
    }
}
