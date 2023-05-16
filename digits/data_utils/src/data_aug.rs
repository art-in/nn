use std::f32::consts::PI;

use dfdx::data::ExactSizeDataset;
use image::{imageops, Rgb, RgbImage};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use moving_least_squares as mls;
use moving_least_squares_image as mls_image;
use rand::Rng;

use crate::{
    data::{MnistDataSet, MnistDataSetKind},
    utils,
};

// Augmented MNIST dataset by applying image deformations to source samples.
pub struct AugmentedMnistDataSet {
    dataset: MnistDataSet,

    // how much times this dataset should be bigger than source dataset.
    // e.g. "1" - dataset will include source samples only, "3" - dataset will be thrice as big
    // (each source sample will have two deformed derivatives)
    aug_ratio: usize,

    deform_cfg: DeformationConfig,
}

pub struct DeformationConfig {
    pub max_mls_grid_density: usize,
    pub max_mls_shift: f32,
    pub max_rotate_degree: (f32, f32),
    pub max_blur: f32,
    pub max_contrast: f32,
    pub max_scale: f32,
    pub max_shift: u32,
}

impl Default for DeformationConfig {
    fn default() -> Self {
        Self {
            max_mls_grid_density: 0,
            max_mls_shift: 0.0,
            max_rotate_degree: (0.0, 0.0),
            max_blur: 0.0,
            max_contrast: 0.0,
            max_scale: 0.0,
            max_shift: 0,
        }
    }
}

impl AugmentedMnistDataSet {
    pub fn new(
        path: &str,
        kind: MnistDataSetKind,
        aug_ratio: usize,
        deform_cfg: DeformationConfig,
    ) -> Self {
        Self {
            dataset: MnistDataSet::new(path, kind),
            aug_ratio,
            deform_cfg,
        }
    }
}

const IMAGE_SIZE: u32 = 28;

impl ExactSizeDataset for AugmentedMnistDataSet {
    type Item<'a> = (Vec<f32>, usize) where Self: 'a;

    fn get(&self, index: usize) -> Self::Item<'_> {
        let original_index = index / self.aug_ratio;
        let (original_image, original_label) = self.dataset.get(original_index);

        if ((index + 1) % self.aug_ratio) == 0 {
            (original_image, original_label)
        } else {
            let original_image_rgb: RgbImage =
                utils::mnist_to_rgb_image(&original_image, IMAGE_SIZE, IMAGE_SIZE);

            let new_image_rgb = deform_image(&original_image_rgb, &self.deform_cfg);
            let new_image = utils::rgb_to_mnist_image(new_image_rgb, IMAGE_SIZE, IMAGE_SIZE);

            (new_image, original_label)
        }
    }

    fn len(&self) -> usize {
        self.dataset.get_labels().len() * self.aug_ratio
    }
}

fn deform_image(original_image_rgb: &RgbImage, cfg: &DeformationConfig) -> RgbImage {
    let mut rng = rand::thread_rng();

    let new_image_rgb = original_image_rgb;

    // MLS deformation
    let new_image_rgb = apply_moving_least_squares_deformation(
        new_image_rgb,
        cfg.max_mls_grid_density,
        cfg.max_mls_shift,
    );

    // rotate
    let new_image_rgb = rotate_about_center(
        &new_image_rgb,
        PI / 180.0 * rng.gen_range(cfg.max_rotate_degree.0..=cfg.max_rotate_degree.1),
        Interpolation::Bicubic,
        Rgb([255, 255, 255]),
    );

    // blur
    let new_image_rgb = imageops::blur(&new_image_rgb, rng.gen_range(0.0..=cfg.max_blur));

    // contrast
    let new_image_rgb = imageops::contrast(&new_image_rgb, rng.gen_range(0.0..=cfg.max_contrast));

    // scale
    let scale_factor = rng.gen_range(1.0 - cfg.max_scale..=1.0 + cfg.max_scale);
    let new_image_rgb = imageops::resize(
        &new_image_rgb,
        (IMAGE_SIZE as f32 * scale_factor) as u32,
        (IMAGE_SIZE as f32 * scale_factor) as u32,
        imageops::FilterType::Gaussian,
    );

    let new_image_rgb = if scale_factor < 1.0 {
        let mut image = RgbImage::new(IMAGE_SIZE, IMAGE_SIZE);
        image.fill(255);
        imageops::overlay(
            &mut image,
            &new_image_rgb,
            (IMAGE_SIZE as f32 * (1.0 - scale_factor) / 2.0) as u32,
            (IMAGE_SIZE as f32 * (1.0 - scale_factor) / 2.0) as u32,
        );
        image
    } else {
        imageops::crop_imm(
            &new_image_rgb,
            (IMAGE_SIZE as f32 * (scale_factor - 1.0) / 2.0) as u32,
            (IMAGE_SIZE as f32 * (scale_factor - 1.0) / 2.0) as u32,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
        .to_image()
    };

    // shift
    {
        // shifting bottom-right only, since negative numbers is not allowed for overlay(),
        // and I don't want to bother with cropping
        let shift_x = rng.gen_range(0..=cfg.max_shift);
        let shift_y = rng.gen_range(0..=cfg.max_shift);

        let mut image = RgbImage::new(IMAGE_SIZE, IMAGE_SIZE);
        image.fill(255);
        imageops::overlay(&mut image, &new_image_rgb, shift_x, shift_y);
        image
    }
}

// "Image Deformation Using Moving Least Squares"
// https://people.engr.tamu.edu/schaefer/research/mls.pdf
fn apply_moving_least_squares_deformation(
    rgb_image: &RgbImage,
    max_grid_density: usize,
    max_shift: f32,
) -> RgbImage {
    let mut rng = rand::thread_rng();

    let grid_density = rng.gen_range(1..=max_grid_density);

    // init control points for deformation grid
    let mut controls_src = Vec::new();
    let mut controls_dst = Vec::new();

    let parts = grid_density + 1;

    for x_part in 0..=parts {
        for y_part in 0..=parts {
            let mut x = IMAGE_SIZE as f32 / parts as f32 * x_part as f32;
            let mut y = IMAGE_SIZE as f32 / parts as f32 * y_part as f32;

            controls_src.push((x, y));

            // randomly shift all control points, except ones on the border
            if x_part > 0 && x_part < parts && y_part > 0 && y_part < parts {
                x += rng.gen_range(-max_shift..=max_shift);
                y += rng.gen_range(-max_shift..=max_shift);
            }

            controls_dst.push((x, y));
        }
    }

    // apply deformation
    let mut deformed_image = mls_image::reverse_dense(
        rgb_image,
        &controls_src,
        &controls_dst,
        mls::deform_similarity,
    );

    // clear black borders which appear after deformation, even though we didn't shift control
    // points on the border
    for x in 0..IMAGE_SIZE {
        for y in 0..IMAGE_SIZE {
            if (x as f32 - max_shift) < 0.0
                || (x as f32 + max_shift) >= IMAGE_SIZE as f32
                || (y as f32 - max_shift) < 0.0
                || (y as f32 + max_shift) >= IMAGE_SIZE as f32
            {
                let white = Rgb([255, 255, 255]);
                deformed_image.put_pixel(x, y, white);
            }
        }
    }

    deformed_image
}
