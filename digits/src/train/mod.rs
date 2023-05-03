use std::time::SystemTime;

use autograd::{network::Network, val::BVal};

use crate::{
    mnist::{images_it::ImagesIt, labels_it::LabelsIt},
    train::plot::plot_losses,
    utils::predict,
};

mod plot;
mod utils;

pub fn train(
    net: &mut Network,
    images_file_path: &str,
    labels_file_path: &str,
    models_dir: &str,
    model_file_name_prefix: &str,
    plots_dir: &str,
    epochs: u32,
    batches: u32,
    batch_size: u32,
    learning_rate: (f64, f64),
    plot_losses_each_nth_batch: Option<u32>,
    serialize_model_each_nth_batch: Option<u32>,
) {
    let images_it = ImagesIt::new(images_file_path);
    let labels_it = LabelsIt::new(labels_file_path);

    assert!(
        batches * batch_size <= images_it.images_count(),
        "not enough images in input stream"
    );

    let mut images_and_labels_it = images_it.zip(labels_it);

    let mut losses = Vec::<f64>::new();
    let mut errors_percents = Vec::<f64>::new();

    for epoch_idx in 0..epochs {
        for batch_idx in 0..batches {
            let batch_start = SystemTime::now();

            // take next batch of images
            let mut batch = Vec::<(Vec<f64>, u8)>::new();

            for _ in 0..batch_size {
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
            let learning_rate = learning_rate.0
                - (learning_rate.0 - learning_rate.1) * batch_idx as f64 / batches as f64;

            for param in net.parameters() {
                let grad = param.borrow().grad;
                param.borrow_mut().d -= learning_rate * grad;
            }

            // log / plot
            if plot_losses_each_nth_batch.is_some()
                && batch_idx > 0
                && batch_idx % plot_losses_each_nth_batch.unwrap() == 0
            {
                plot_losses(&losses, &errors_percents, plots_dir);
            }

            if serialize_model_each_nth_batch.is_some()
                && batch_idx > 0
                && batch_idx % serialize_model_each_nth_batch.unwrap() == 0
            {
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

        if plot_losses_each_nth_batch.is_some() {
            plot_losses(&losses, &errors_percents, plots_dir);
        }

        if serialize_model_each_nth_batch.is_some() {
            net.serialize_to_file(models_dir, model_file_name_prefix);
        }
    }
}
