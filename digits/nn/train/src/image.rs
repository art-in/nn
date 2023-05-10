use image::{ImageBuffer, Rgb, RgbImage};

pub fn create_image(image: &[f64], width: u32, height: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut rgb_image = RgbImage::new(width, height);

    for (idx, pixel) in image.iter().enumerate() {
        let x = idx as u32 % width;
        let y = idx as u32 / width;
        let pixel = *pixel * 255.0;
        let pixel = pixel.min(255.0);
        let pixel = 255 - pixel.max(0.0) as u8;
        rgb_image.put_pixel(x, y, Rgb([pixel, pixel, pixel]));
    }

    rgb_image
}
