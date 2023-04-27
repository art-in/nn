use rand_distr::{Distribution, Normal};

// generates random number with normal distribution in range (-1, 1)
pub fn gen_rand_normal() -> f64 {
    let normal = Normal::new(0.0, 0.15).unwrap();
    normal.sample(&mut rand::thread_rng())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_rand_normal() {
        for _ in 0..1_000_000 {
            let val = gen_rand_normal();

            if val.abs() > 0.9 {
                panic!();
            }
        }
    }
}
