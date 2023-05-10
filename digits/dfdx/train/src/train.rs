use std::time::Instant;

use dfdx::optim::{Momentum, Sgd, SgdConfig};
use indicatif::ProgressIterator;
use mnist::*;
use rand::prelude::{SeedableRng, StdRng};

use dfdx::prelude::*;
use dfdx::{data::*, tensor::AutoDevice};

use crate::model_type::ModelBuild;
use crate::test::{self};

struct MnistTrainSet(Mnist);

impl MnistTrainSet {
    fn new(path: &str) -> Self {
        Self(MnistBuilder::new().base_path(path).finalize())
    }
}

impl ExactSizeDataset for MnistTrainSet {
    type Item<'a> = (Vec<f32>, usize) where Self: 'a;
    fn get(&self, index: usize) -> Self::Item<'_> {
        let mut img_data: Vec<f32> = Vec::with_capacity(784);
        let start = 784 * index;
        img_data.extend(
            self.0.trn_img[start..start + 784]
                .iter()
                .map(|x| *x as f32 / 255.0),
        );
        (img_data, self.0.trn_lbl[index] as usize)
    }
    fn len(&self) -> usize {
        self.0.trn_lbl.len()
    }
}

const EPOCHS: i32 = 30;
const BATCH_SIZE: usize = 32;

const MNIST_PATH: &str = "../../mnist/";

pub fn train(dev: &AutoDevice, model: &mut ModelBuild) {
    // ftz substantially improves performance
    dfdx::flush_denormals_to_zero();

    let mut rng = StdRng::seed_from_u64(0);

    // initialize model, gradients, and optimizer
    let mut grads = model.alloc_grads();

    // initialize dataset
    let dataset = MnistTrainSet::new(MNIST_PATH);
    println!("Found {:?} training images", dataset.len());

    let preprocess = |(img, lbl): <MnistTrainSet as ExactSizeDataset>::Item<'_>| {
        let mut one_hotted = [0.0; 10];
        one_hotted[lbl] = 1.0;
        (
            dev.tensor_from_vec(img, (Const::<1>, Const::<28>, Const::<28>)),
            dev.tensor(one_hotted),
        )
    };

    let mut opt: Sgd<ModelBuild, f32, AutoDevice> = Sgd::new(
        model,
        SgdConfig {
            lr: 0.01,
            momentum: Some(Momentum::Classic(0.9)),
            weight_decay: None,
        },
    );

    for i_epoch in 0..EPOCHS {
        let mut total_epoch_loss = 0.0;
        let mut num_batches = 0;
        let start = Instant::now();
        for (img, lbl) in dataset
            .shuffled(&mut rng)
            .map(preprocess)
            .batch_exact(Const::<BATCH_SIZE>)
            .collate()
            .stack()
            .progress()
        {
            let logits = model.forward_mut(img.traced(grads));
            let loss = cross_entropy_with_logits_loss(logits, lbl);

            total_epoch_loss += loss.array();
            num_batches += 1;

            grads = loss.backward();
            opt.update(model, &grads).unwrap();
            model.zero_grads(&mut grads);
        }
        let dur = Instant::now() - start;

        println!(
            "Epoch {i_epoch} in {:?} ({:.3} batches/s): avg sample loss {:.5}",
            dur,
            num_batches as f32 / dur.as_secs_f32(),
            BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32
        );

        test::test(model);
    }

    model
        .save("./models/mnist.npz")
        .expect("failed to save model");
}
