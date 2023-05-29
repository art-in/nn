use std::time::Instant;
use std::usize;

use dfdx::data::{IteratorBatchExt, IteratorCollateExt, IteratorStackExt};
use dfdx::optim::{Adam, AdamConfig};
use indicatif::ProgressIterator;

use dfdx::prelude::*;
use dfdx::tensor::AutoDevice;

use crate::game_iterator::{GameIterator, IMAGE_SIZE, PREDICTION_POSITIONS};
use crate::model_type::ModelBuild;
use crate::{test, MODEL_PATH};

const EPOCHS: i32 = 50;
const BATCH_SIZE: usize = 32;
const GAME_STEPS: usize = 10_000;

pub fn train(device: &AutoDevice, model: &mut ModelBuild) {
    // ftz substantially improves performance
    dfdx::flush_denormals_to_zero();

    println!(
        "start training. model params: {}",
        model.num_trainable_params()
    );

    let mut grads = model.alloc_grads();
    let mut opt = Adam::new(model, AdamConfig::default());

    let tenzorify = |(image, label)| {
        (
            device.tensor_from_vec(image, (Const::<IMAGE_SIZE>,)),
            device.tensor_from_vec(label, (Const::<PREDICTION_POSITIONS>,)),
        )
    };

    for epoch_idx in 0..EPOCHS {
        let epoch_start = Instant::now();

        let game_iterator = GameIterator::new(GAME_STEPS);

        let mut epoch_loss = 0.0;
        let mut epoch_batches_count = 0;

        for (image, label) in game_iterator
            .map(tenzorify)
            .batch_exact(Const::<BATCH_SIZE>)
            .collate()
            .stack()
            .progress()
        {
            let logits = model.forward_mut(image.traced(grads));
            let loss = cross_entropy_with_logits_loss(logits, label);

            epoch_loss += loss.array();
            epoch_batches_count += 1;

            grads = loss.backward();
            opt.update(model, &grads).unwrap();
            model.zero_grads(&mut grads);
        }

        let epoch_duration = Instant::now().duration_since(epoch_start);
        let errors_percent = test::test(device, model, false);

        println!(
            "epoch {epoch_idx} in {}s ({:.3} batches/s): avg sample loss {:.5}, errors = {:.2}%",
            epoch_duration.as_secs(),
            epoch_batches_count as f32 / epoch_duration.as_secs_f32(),
            BATCH_SIZE as f32 * epoch_loss / epoch_batches_count as f32,
            errors_percent
        );

        model.save(MODEL_PATH).expect("failed to save model");
    }
}
