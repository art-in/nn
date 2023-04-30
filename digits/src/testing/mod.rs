use autograd::network::Network;

use crate::{
    image::create_image,
    mnist::{images_it::ImagesIt, labels_it::LabelsIt},
    utils::predict,
};

pub fn test(net: &Network) {
    let images_it = ImagesIt::new("digits/data/t10k-images-idx3-ubyte");
    let labels_it = LabelsIt::new("digits/data/t10k-labels-idx1-ubyte");

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
        if is_error {
            let image = create_image(&image, image_width, image_height);
            image
                .save(format!("digits/images/error-{idx}-{label}.png"))
                .expect("failed to save image");
        }
    }

    println!(
        "count: {count}, errors: {errors}, errors_percent: {}",
        errors as f64 / count as f64
    );
}
