use num::ToPrimitive;
// use rayon::iter::IntoParallelIterator;
use std::ffi::{c_float, c_int};
extern crate test;

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

#[cfg(test)]
mod tests {
    use crate::utils::make_random_float;

    use super::*;
    use test::Bencher;

    const B: usize = 8;
    const T: usize = 1024;
    const C: usize = 768;
    const OC: usize = 768;
    const SQRT_BLOCK_SIZE: usize = 16;

    fn matmul_forward_cpu(out: &mut [f32], inp: &[f32], weight: &[f32], bias: &[f32]) {
        // OC is short for "output channels"
        // inp is (B,T,C), weight is (OC, C), bias is (OC)
        // out will be (B,T,OC)

        for b in 0..B {
            for t in 0..T {
                let out_bt = b * T * OC + t * OC;
                let inp_bt = b * T * C + t * C;
                for o in 0..OC {
                    let mut val = if bias.len() > 0 { bias[o] } else { 0.0 };
                    let wrow = o * C;
                    for i in 0..C {
                        val += inp[inp_bt + i] * weight[wrow + i];
                    }
                    out[out_bt + o] = val;
                }
            }
        }
    }

    #[test]
    fn test_matmul_forward_kernel1() {
        let mut inp = make_random_float(B * T * C);

        let mut out: Vec<f32> = Vec::with_capacity(B * T * OC);

        let mut weight = make_random_float(OC * C);

        let mut bias = make_random_float(OC);

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
                SQRT_BLOCK_SIZE.to_i32().unwrap(),
            );
        };
        // getting result from cuda
        let out_gpu = unsafe { std::slice::from_raw_parts(out.as_mut_ptr(), B * T * OC) };

        // computing result on cpu for comparison
        let mut out_cpu: Vec<f32> = vec![0.0; B * T * OC];
        matmul_forward_cpu(&mut out_cpu, &mut inp, &mut weight, &mut bias);

        assert_eq!(out_cpu.len(), out_gpu.to_vec().len());
        assert_eq!(out_cpu, out_gpu.to_vec());
    }

    #[bench]
    fn bench_kernel1(bencher: &mut Bencher) {
        let mut inp = make_random_float(B * T * C);

        let mut out: Vec<f32> = Vec::with_capacity(B * T * OC);

        let mut weight = make_random_float(OC * C);

        let mut bias = make_random_float(OC);

        bencher.iter(|| {
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
                    SQRT_BLOCK_SIZE.to_i32().unwrap(),
                );
            };
        })
    }

    #[bench]
    fn bench_kernel2(bencher: &mut Bencher) {
        let mut inp = make_random_float(B * T * C);

        let mut out: Vec<f32> = Vec::with_capacity(B * T * OC);

        let mut weight = make_random_float(OC * C);

        let mut bias = make_random_float(OC);

        bencher.iter(|| {
            unsafe {
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
                    SQRT_BLOCK_SIZE.to_i32().unwrap(),
                );
            };
        })
    }

    #[bench]
    fn bench_kernel3(bencher: &mut Bencher) {
        let mut inp = make_random_float(B * T * C);

        let mut out: Vec<f32> = Vec::with_capacity(B * T * OC);

        let mut weight = make_random_float(OC * C);

        let mut bias = make_random_float(OC);

        bencher.iter(|| {
            unsafe {
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
                    SQRT_BLOCK_SIZE.to_i32().unwrap(),
                );
            };
        })
    }
}
