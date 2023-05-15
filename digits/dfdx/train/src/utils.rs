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
