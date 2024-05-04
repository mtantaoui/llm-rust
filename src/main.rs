use std::ffi::{c_float, c_int};

use num::ToPrimitive;

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

}

fn main() {
    let B = 2;
    let T = 2;
    let C = 2;
    let OC = 2;

    let sqrt_block_size = 16;

    let mut inp: Vec<f32> = Vec::new();
    inp = vec![1.0; B * T * C];

    let mut out: Vec<f32> = Vec::new();
    out = vec![2.0; B * T * OC];

    let mut weight: Vec<f32> = Vec::with_capacity(OC * C);
    weight = vec![3.0; OC * C];

    let mut bias: Vec<f32> = Vec::with_capacity(OC);
    bias = vec![4.0; OC];

    unsafe {
        matmul_forward_cuda(
            1,
            out.as_mut_ptr(),
            inp.as_mut_ptr(),
            weight.as_mut_ptr(),
            bias.as_mut_ptr(),
            B.to_i32().unwrap(),
            T.to_i32().unwrap(),
            C.to_i32().unwrap(),
            OC.to_i32().unwrap(),
            sqrt_block_size,
        );

        matmul_forward_cuda(
            2,
            out.as_mut_ptr(),
            inp.as_mut_ptr(),
            weight.as_mut_ptr(),
            bias.as_mut_ptr(),
            B.to_i32().unwrap(),
            T.to_i32().unwrap(),
            C.to_i32().unwrap(),
            OC.to_i32().unwrap(),
            sqrt_block_size,
        );
        matmul_forward_cuda(
            3,
            out.as_mut_ptr(),
            inp.as_mut_ptr(),
            weight.as_mut_ptr(),
            bias.as_mut_ptr(),
            B.to_i32().unwrap(),
            T.to_i32().unwrap(),
            C.to_i32().unwrap(),
            OC.to_i32().unwrap(),
            sqrt_block_size,
        );
    };
    // cuda();
}
