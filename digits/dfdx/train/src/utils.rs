use image::{Rgb, RgbImage};

pub fn argmax(output: &[f32]) -> usize {
    let mut max_idx = 0usize;
    let mut max_out = output[0];

    for (idx, out) in output.iter().enumerate() {
        if *out > max_out {
            max_idx = idx;
            max_out = *out;
        }
    }

    max_idx
}

pub fn mnist_to_rgb_image(mnist_image: &[f32], width: u32, height: u32) -> RgbImage {
    let mut rgb_image = RgbImage::new(width, height);

    for (pixel_idx, mnist_pixel) in mnist_image.iter().enumerate() {
        assert!(
            *mnist_pixel >= 0.0 && *mnist_pixel <= 1.0,
            "pixel not in range"
        );

        let x = pixel_idx as u32 % width;
        let y = pixel_idx as u32 / width;

        let rgb_pixel = 255 - (*mnist_pixel * 255.0) as u8;
        rgb_image.put_pixel(x, y, Rgb([rgb_pixel, rgb_pixel, rgb_pixel]));
    }

    rgb_image
}

pub fn rgb_to_mnist_image(rgb_image: RgbImage, width: u32, height: u32) -> Vec<f32> {
    let pixels_count = width * height;

    let mut mnist_image = Vec::new();
    mnist_image.resize(pixels_count as usize, 0.0);

    for x in 0..width {
        for y in 0..height {
            let pixel_idx = y * width + x;
            let rgb_pixel = rgb_image.get_pixel(x, y);

            mnist_image[pixel_idx as usize] = 1.0 - rgb_pixel[0] as f32 / 255.0;
        }
    }

    mnist_image
}
