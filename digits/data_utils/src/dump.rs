use data_utils::{data::MnistDataSetKind, data_aug::AugmentedMnistDataSet};
use dfdx::data::ExactSizeDataset;
use indicatif::ProgressIterator;

use crate::utils;

#[allow(dead_code)]
pub fn dump_dataset_images(mnist_path: &str) {
    let dataset = AugmentedMnistDataSet::new(mnist_path, MnistDataSetKind::Train, 2);

    let mut rng = rand::thread_rng();
    for (idx, (image, label)) in dataset.shuffled(&mut rng).enumerate().progress() {
        utils::mnist_to_rgb_image(&image, 28, 28)
            .save(format!("./dump/{idx}-{label}.png"))
            .expect("failed to save image");
    }
}
