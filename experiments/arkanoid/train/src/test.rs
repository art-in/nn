use dfdx::{
    prelude::{Module, NumParams},
    shapes::Const,
    tensor::{AutoDevice, TensorFromVec},
};
use indicatif::ProgressIterator;

use crate::{
    game_iterator::{GameIterator, IMAGE_SIZE},
    model_type::ModelBuild,
    utils,
};

const GAME_STEPS: usize = 1_000;

pub fn test(device: &AutoDevice, model: &ModelBuild, is_log: bool) -> f32 {
    if is_log {
        println!(
            "start testing. model params: {}",
            model.num_trainable_params()
        );
    }

    let game_iterator = GameIterator::new(GAME_STEPS);
    let tensorify = |(image, label)| (device.tensor_from_vec(image, (Const::<IMAGE_SIZE>,)), label);
    let mut errors = 0;

    for (image, label) in game_iterator.map(tensorify).progress() {
        let logits = model.forward(image);

        let predicted_label = utils::argmax(&logits.as_vec()) as u32;
        let expected_label = utils::argmax(&label) as u32;

        errors += (predicted_label != expected_label) as i32;
    }

    let errors_percent = (errors as f32 / GAME_STEPS as f32) * 100.0;

    if is_log {
        println!(
            "steps: {}, errors: {errors}, error_percent: {:.2}%",
            GAME_STEPS, errors_percent
        );
    }

    errors_percent
}
