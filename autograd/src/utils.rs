use std::{
    io::{Read, Write},
    path::Path,
};

use rand_distr::{Distribution, Normal};

// generates random number with normal distribution
pub fn gen_rand_normal(deviation: f64) -> f64 {
    let normal = Normal::new(0.0, deviation).unwrap();
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

fn get_model_file_name(prefix: &str, layers_sizes: &Vec<usize>) -> String {
    let mut res = layers_sizes.iter().fold(prefix.to_string(), |mut res, sz| {
        res += "-";
        res += &sz.to_string();
        res
    });

    res += ".nm";
    res
}

pub fn get_model_file_path(dir: &str, file_name_prefix: &str, layers_sizes: &Vec<usize>) -> String {
    String::from(
        Path::new(dir)
            .join(get_model_file_name(file_name_prefix, &layers_sizes))
            .to_str()
            .unwrap(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_rand_normal() {
        for _ in 0..1_000_000 {
            let val = gen_rand_normal(0.15);

            if val.abs() > 0.9 {
                panic!();
            }
        }
    }
}
