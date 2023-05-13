use dfdx::{
    data::ExactSizeDataset,
    prelude::{Module, NumParams},
    shapes::Rank3,
    tensor::{Cpu, Tensor, TensorFrom},
};
use indicatif::ProgressIterator;

use crate::{
    data::MnistDataSetKind, data_aug::AugmentedMnistDataSet, model_type::ModelBuild, utils,
};

pub fn test(model: &ModelBuild, is_log: bool) -> f32 {
    let device = Cpu::default();

    let dataset = AugmentedMnistDataSet::new(MnistDataSetKind::Test, 2);

    if is_log {
        println!(
            "start testing. model params: {}, dataset size: {}",
            model.num_trainable_params(),
            dataset.len()
        );
    }

    let mut errors = 0;

    for (image, label) in dataset.iter().progress() {
        let input: Tensor<Rank3<1, 28, 28>, f32, _> = device.tensor(image);
        let output = model.forward(input);
        let predicted_label = utils::argmax(&output.as_vec());
        errors += (predicted_label != label) as i32;
    }

    let errors_percent = (errors as f32 / dataset.len() as f32) * 100.0;

    if is_log {
        println!(
            "images: {}, errors: {errors}, error_percent: {:.2}%",
            dataset.len(),
            errors_percent
        );
    }

    errors_percent
}
