use std::time::{Instant, SystemTime};

use chrono::{DateTime, Utc};
use dfdx::optim::{Momentum, Sgd, SgdConfig};
use indicatif::ProgressIterator;
use rand::prelude::{SeedableRng, StdRng};

use dfdx::prelude::*;
use dfdx::{data::*, tensor::AutoDevice};

use crate::data::MnistDataSetKind;
use crate::data_aug::AugmentedMnistDataSet;
use crate::model_type::ModelBuild;
use crate::test::{self};

const EPOCHS: i32 = 10;
const BATCH_SIZE: usize = 32;
const IMAGE_SIZE: usize = 28;

pub fn train(dev: &AutoDevice, model: &mut ModelBuild, model_path: &str) {
    // ftz substantially improves performance
    dfdx::flush_denormals_to_zero();

    let dataset = AugmentedMnistDataSet::new(MnistDataSetKind::Train, 2);
    println!(
        "start training. time: {}, found {:?} training images",
        DateTime::<Utc>::from(SystemTime::now()).to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        dataset.len()
    );

    let mut rng = StdRng::seed_from_u64(0);
    let mut grads = model.alloc_grads();

    let mut opt: Sgd<ModelBuild, f32, AutoDevice> = Sgd::new(
        model,
        SgdConfig {
            lr: 0.01,
            momentum: Some(Momentum::Classic(0.9)),
            weight_decay: None,
        },
    );

    let tenzorify = |(image, label)| {
        let mut one_hotted = [0.0; 10];
        one_hotted[label] = 1.0;
        (
            dev.tensor_from_vec(
                image,
                (Const::<1>, Const::<IMAGE_SIZE>, Const::<IMAGE_SIZE>),
            ),
            dev.tensor(one_hotted),
        )
    };

    for epoch_idx in 0..EPOCHS {
        let epoch_start = Instant::now();

        let mut total_epoch_loss = 0.0;
        let mut batches_count = 0;

        for (image, label) in dataset
            .shuffled(&mut rng)
            .map(tenzorify)
            .batch_exact(Const::<BATCH_SIZE>)
            .collate()
            .stack()
            .progress()
        {
            let logits = model.forward_mut(image.traced(grads));
            let loss = cross_entropy_with_logits_loss(logits, label);

            total_epoch_loss += loss.array();
            batches_count += 1;

            grads = loss.backward();
            opt.update(model, &grads).unwrap();
            model.zero_grads(&mut grads);
        }

        let duration = Instant::now().duration_since(epoch_start);

        let errors_percent = test::test(model, false);

        println!(
            "epoch {epoch_idx} in {}s ({:.3} batches/s): avg sample loss {:.5}, errors {:.2}%",
            duration.as_secs(),
            batches_count as f32 / duration.as_secs_f32(),
            BATCH_SIZE as f32 * total_epoch_loss / batches_count as f32,
            errors_percent
        );

        model.save(model_path).expect("failed to save model");
    }
}
