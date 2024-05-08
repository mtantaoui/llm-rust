use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};

pub fn make_random_float(n: usize) -> Vec<f32> {
    // Initialize a random number generator
    let mut rng = StdRng::from_entropy();

    (0..n).map(|_| rng.gen_range(-1.0..=1.0)).collect()
}

pub fn make_random_float_01(n: usize) -> Vec<f32> {
    // Initialize a random number generator
    let mut rng = thread_rng();

    (0..n).map(|_| rng.gen::<f32>()).collect()
}
