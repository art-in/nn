use autograd::network::Network;

use crate::{
    image::create_image,
    mnist::{images_it::ImagesIt, labels_it::LabelsIt},
    utils::predict,
};

pub fn test(
    net: &Network,
    images_file_path: &str,
    labels_file_path: &str,
    failed_images_dir: &str,
    save_failed_images: bool,
) {
    let images_it = ImagesIt::new(images_file_path);
    let labels_it = LabelsIt::new(labels_file_path);

    let image_width = images_it.image_width();
    let image_height = images_it.image_height();

    let images_and_labels_it = images_it.zip(labels_it);

    let mut errors = 0;
    let mut count = 0;

    for (idx, (image, label)) in images_and_labels_it.enumerate() {
        let output = net.forward(&image);

        let is_error = label != predict(&output);
        errors += if is_error { 1 } else { 0 };
        count += 1;

        // save failed image
        if is_error && save_failed_images {
            let image = create_image(&image, image_width, image_height);
            image
                .save(format!("{failed_images_dir}/error-{idx}-{label}.png"))
                .expect("failed to save image");
        }
    }

    println!(
        "count: {count}, errors: {errors}, errors_percent: {}",
        errors as f64 / count as f64
    );
}
