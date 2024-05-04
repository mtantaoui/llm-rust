extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .cpp(true)
        .flag("-cudart=shared")
        .flag("--use_fast_math")
        .flag("-lcublas")
        .flag("-lcublasLt")
        .files(&[
            "cuda/matmul_forward/matmul_forward_ffi.cpp",
            "cuda/matmul_forward/matmul_forward.cu",
        ])
        .compile("kernels.out");

    println!("cargo:rustc-env=LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64");

    println!("cargo:rustc-link-search=native=/usr/local/cuda-12.3/lib64");
    println!("cargo:rustc-link-search=/usr/local/cuda-12.3/lib64");

    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
}
