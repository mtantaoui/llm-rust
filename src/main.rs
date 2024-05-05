use std::ffi::{c_float, c_int, c_long};

use num::ToPrimitive;
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};

fn make_random_float(n: i64) -> Vec<f32> {
    // Initialize a random number generator
    let mut rng = StdRng::from_entropy();

    (0..n).map(|_| rng.gen_range(-1.0..=1.0)).collect()
}

fn make_random_float_01(n: i64) -> Vec<f32> {
    // Initialize a random number generator
    let mut rng = thread_rng();

    (0..n).map(|_| rng.gen::<f32>()).collect()
}

extern "C" {

    fn matmul_forward_cuda(
        kernel_num: c_int,
        out: *mut c_float,
        inp: *mut c_float,
        weight: *mut c_float,
        bias: *mut c_float,
        B: c_int,
        T: c_int,
        C: c_int,
        OC: c_int,
        sqrt_block_size: c_int,
    );

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
    let num_parameters: i64 = 1048576;
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
            num_parameters,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
        )
    }
}

fn matmul_forward_test() {
    let b = 8;
    let t = 1024;
    let c = 768;
    let oc = 768 * 4;

    let sqrt_block_size = 16;

    let mut inp: Vec<f32> = vec![1.0; b * t * c];

    let mut out: Vec<f32> = vec![2.0; b * t * oc];

    let mut weight: Vec<f32> = vec![3.0; oc * c];

    let mut bias: Vec<f32> = vec![4.0; oc];

    unsafe {
        matmul_forward_cuda(
            1,
            out.as_mut_ptr(),
            inp.as_mut_ptr(),
            weight.as_mut_ptr(),
            bias.as_mut_ptr(),
            b.to_i32().unwrap(),
            t.to_i32().unwrap(),
            c.to_i32().unwrap(),
            oc.to_i32().unwrap(),
            sqrt_block_size,
        );

        matmul_forward_cuda(
            2,
            out.as_mut_ptr(),
            inp.as_mut_ptr(),
            weight.as_mut_ptr(),
            bias.as_mut_ptr(),
            b.to_i32().unwrap(),
            t.to_i32().unwrap(),
            c.to_i32().unwrap(),
            oc.to_i32().unwrap(),
            sqrt_block_size,
        );
        matmul_forward_cuda(
            3,
            out.as_mut_ptr(),
            inp.as_mut_ptr(),
            weight.as_mut_ptr(),
            bias.as_mut_ptr(),
            b.to_i32().unwrap(),
            t.to_i32().unwrap(),
            c.to_i32().unwrap(),
            oc.to_i32().unwrap(),
            sqrt_block_size,
        );
    };
}

fn main() {
    adamw_test();
    // matmul_forward_test();
}
