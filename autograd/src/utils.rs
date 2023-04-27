use std::io::{Read, Write};

use rand_distr::{Distribution, Normal};

// generates random number with normal distribution in range (-1, 1)
pub fn gen_rand_normal() -> f64 {
    let normal = Normal::new(0.0, 0.15).unwrap();
    normal.sample(&mut rand::thread_rng())
}

pub fn write_u32<T: Write>(writer: &mut T, n: u32) {
    let wrote_bytes = writer.write(&n.to_ne_bytes()).expect("failed to write");
    assert_eq!(wrote_bytes, 4);
}

pub fn read_u32<T: Read>(reader: &mut T) -> u32 {
    let mut buf: [u8; 4] = [0; 4];
    reader.read_exact(&mut buf).expect("failed to read");
    u32::from_ne_bytes(buf)
}

pub fn write_f64<T: Write>(writer: &mut T, n: f64) {
    let wrote_bytes = writer.write(&n.to_ne_bytes()).expect("failed to write");
    assert_eq!(wrote_bytes, 8);
}

pub fn read_f64<T: Read>(reader: &mut T) -> f64 {
    let mut buf: [u8; 8] = [0; 8];
    reader.read_exact(&mut buf).expect("failed to read");
    f64::from_ne_bytes(buf)
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
