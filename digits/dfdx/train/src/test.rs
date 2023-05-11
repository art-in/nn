use dfdx::{
    data::ExactSizeDataset,
    prelude::Module,
    shapes::Rank3,
    tensor::{Cpu, Tensor, TensorFrom},
};
use indicatif::ProgressIterator;

use crate::{
    data::{MnistDataSet, MnistDataSetKind},
    model_type::ModelBuild,
    utils,
};

pub fn test(model: &ModelBuild) -> f32 {
    let device = Cpu::default();

    let dataset = MnistDataSet::new(MnistDataSetKind::Test);

    let mut errors = 0;

    for (image, label) in dataset.iter().progress() {
        let input: Tensor<Rank3<1, 28, 28>, f32, _> = device.tensor(image);
        let output = model.forward(input);
        let predicted_label = utils::argmax(&output.as_vec());
        errors += (predicted_label != label) as i32;
    }

    let errors_percent = (errors as f32 / dataset.len() as f32) * 100.0;

    println!(
        "images: {}, errors: {errors}, error_percent: {:.2}%",
        dataset.len(),
        errors_percent
    );

    errors_percent
}
