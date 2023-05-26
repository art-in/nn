use crate::inference::PX_SIZE;

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

pub fn scale_image(
    source_image: &[u8],
    source_size: (usize, usize),
    target_size: (usize, usize),
) -> Vec<u8> {
    assert_eq!(
        source_image.len(),
        source_size.0 * source_size.1 * PX_SIZE,
        "invalid image size"
    );

    let mut res = Vec::new();

    let scale_x = source_size.0 as f64 / target_size.0 as f64;
    let scale_y = source_size.1 as f64 / target_size.1 as f64;

    for target_y in 0..target_size.1 {
        for target_x in 0..target_size.0 {
            let source_x = (target_x as f64 * scale_x) as usize;
            let source_y = (target_y as f64 * scale_y) as usize;

            let source_px_idx = source_y * source_size.0 + source_x;

            res.push(source_image[source_px_idx * PX_SIZE]);
            res.push(source_image[source_px_idx * PX_SIZE + 1]);
            res.push(source_image[source_px_idx * PX_SIZE + 2]);
            res.push(source_image[source_px_idx * PX_SIZE + 3]);
        }
    }

    res
}

pub fn map_prediction_position_to_game_position(pos: u32, positions_count: usize) -> f64 {
    let step = 1.0 / positions_count as f64;
    pos as f64 * step
}
