use data_utils::data::{MnistDataSet, MnistDataSetKind};
use dfdx::data::ExactSizeDataset;
use indicatif::ProgressIterator;

use crate::MNIST_PATH;

const IMAGE_SIZE: usize = 28;

#[allow(dead_code)]
pub fn print_avg_vertical_padding() {
    let dataset: MnistDataSet = MnistDataSet::new(MNIST_PATH, MnistDataSetKind::Train);

    let mut vertical_padding = 0;
    let mut count = 0;

    for (image, _label) in dataset.iter().progress() {
        let bounding_rect = get_image_bounding_rect(&image);

        vertical_padding += bounding_rect.topmost_y;
        vertical_padding += IMAGE_SIZE - bounding_rect.bottommost_y;

        count += 2;
    }

    println!(
        "avg vertical padding: {}",
        vertical_padding as f32 / count as f32
    );
}

struct BoundingRect {
    topmost_y: usize,
    bottommost_y: usize,
}

fn get_image_bounding_rect(image: &[f32]) -> BoundingRect {
    let mut topmost_y = IMAGE_SIZE - 1;
    let mut bottommost_y = 0;

    for (idx, pixel) in image.iter().enumerate() {
        let y = idx / IMAGE_SIZE;

        if *pixel > 0.0 {
            topmost_y = topmost_y.min(y);
            bottommost_y = bottommost_y.max(y);
        }
    }

    BoundingRect {
        topmost_y,
        bottommost_y,
    }
}
