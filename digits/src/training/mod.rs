use std::time::SystemTime;

use autograd::{network::Network, val::BVal};

use crate::{
    mnist::{images_it::ImagesIt, labels_it::LabelsIt},
    training::plot::plot_losses,
    utils::predict,
};

mod plot;
mod utils;

const EPOCHS: u32 = 1;
const BATCHES: u32 = 6000;
const BATCH_SIZE: u32 = 10;

const LEARNING_RATE_START: f64 = 0.01;
const LEARNING_RATE_END: f64 = 0.01;

const PLOT_LOSSES_EACH_NTH_BATCH: u32 = 50;
const SERIALIZE_MODEL_EACH_NTH_BATCH: u32 = 1000;

pub fn train(net: &mut Network, models_dir: &str, model_file_name_prefix: &str) {
    let images_it = ImagesIt::new("digits/data/train-images-idx3-ubyte");
    let labels_it = LabelsIt::new("digits/data/train-labels-idx1-ubyte");

    assert!(
        BATCHES * BATCH_SIZE <= images_it.images_count(),
        "not enough images in input stream"
    );

    let mut images_and_labels_it = images_it.zip(labels_it);

    let mut losses = Vec::<f64>::new();
    let mut errors_percents = Vec::<f64>::new();

    for epoch_idx in 0..EPOCHS {
        for batch_idx in 0..BATCHES {
            let batch_start = SystemTime::now();

            // take next batch of images
            let mut batch = Vec::<(Vec<f64>, u8)>::new();

            for _ in 0..BATCH_SIZE {
                batch.push(
                    images_and_labels_it
                        .next()
                        .expect("failed to read next image"),
                );
            }

            // forward
            let mut batch_loss = BVal::new(0.0);
            let mut batch_errors = 0;

            for (image, label) in &batch {
                let output = net.forward(&image);
                let expected = utils::label_to_outputs(*label);

                let loss = utils::calc_prediction_loss(&output, &expected);
                batch_loss = &batch_loss + &loss;

                if *label != predict(&output) {
                    batch_errors += 1;
                }
            }

            let batch_errors_percent = batch_errors as f64 / batch.len() as f64;

            losses.push(batch_loss.borrow().d);
            errors_percents.push(batch_errors_percent);

            // backward
            net.reset_grad();
            batch_loss.borrow_mut().grad = 1.0;
            batch_loss.backward();

            // update
            let learning_rate = LEARNING_RATE_START
                - (LEARNING_RATE_START - LEARNING_RATE_END) * batch_idx as f64 / BATCHES as f64;

            for param in &net.parameters() {
                let grad = param.borrow().grad;
                param.borrow_mut().d -= learning_rate * grad;
            }

            // log / plot
            if (batch_idx + 1) % PLOT_LOSSES_EACH_NTH_BATCH == 0 {
                plot_losses(&losses, &errors_percents);
            }

            if (batch_idx + 1) % SERIALIZE_MODEL_EACH_NTH_BATCH == 0 {
                net.serialize_to_file(models_dir, model_file_name_prefix);
            }

            let batch_duration = SystemTime::now().duration_since(batch_start).unwrap();

            println!(
                "epoch = {epoch_idx}, \
                batch = {batch_idx}, \
                duration = {duration}ms, \
                rate = {learning_rate:.4}, \
                loss = {loss:.4}, \
                errors = {errors_percent}%",
                duration = batch_duration.as_millis(),
                loss = batch_loss.borrow().d,
                errors_percent = batch_errors_percent * 100 as f64
            );
        }
    }
}
