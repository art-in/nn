use dfdx::{
    prelude::Module,
    shapes::Const,
    tensor::{AutoDevice, TensorFromVec},
};

use crate::{model::NETWORK, utils};

pub const PX_SIZE: usize = 4;
pub const IMAGE_PX_SIZE: usize = 100;
pub const IMAGE_SIZE: usize = IMAGE_PX_SIZE * IMAGE_PX_SIZE * PX_SIZE;
pub const PREDICTION_POSITIONS: usize = 10;

pub fn predict_platform_position_x(image: &[u8], size: (usize, usize)) -> f64 {
    let device = AutoDevice::default();

    let image = utils::scale_image(image, size, (IMAGE_PX_SIZE, IMAGE_PX_SIZE));
    assert_eq!(image.len(), IMAGE_SIZE, "invalid image size");

    let image = image.iter().map(|n| *n as f32 / 255.0).collect();
    let image = device.tensor_from_vec(image, (Const::<IMAGE_SIZE>,));

    let logits = NETWORK.forward(image);

    utils::map_prediction_position_to_game_position(
        utils::argmax(&logits.as_vec()) as u32,
        PREDICTION_POSITIONS,
    )
}
