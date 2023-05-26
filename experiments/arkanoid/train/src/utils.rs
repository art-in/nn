pub fn argmax(output: &[f32]) -> usize {
    let mut max_idx = 0usize;
    let mut max_out = output[0];

    for (idx, out) in output.iter().enumerate() {
        if *out > max_out {
            max_idx = idx;
            max_out = *out;
        }
    }

    max_idx
}

#[allow(dead_code)]
pub fn onehot_array<const TSIZE: usize>(one_idx: u32) -> [f32; TSIZE] {
    let mut arr = [0.0_f32; TSIZE];
    arr[one_idx as usize] = 1.0;
    arr
}

pub fn onehot(size: usize, one_idx: u32) -> Vec<f32> {
    let mut vec = Vec::new();
    vec.resize(size as usize, 0.0);
    vec[one_idx as usize] = 1.0;
    vec
}
