use dfdx::{
    prelude::Module,
    shapes::Rank3,
    tensor::{AutoDevice, Tensor, TensorFrom},
};

use crate::model::NETWORK;

const IMAGE_SIZE: usize = 28;

// image represented as series of pixels, where each each pixel is a number in range [-1, 1]
pub fn infer(image: &[f64]) -> Vec<f64> {
    // model was trained on pixels in range [0, 1], so convert pixels [-1, 1] to [0, 1]
    let image: Vec<f32> = image.iter().map(|n| *n as f32 / 2.0 + 0.5).collect();

    let device = AutoDevice::default();
    let input: Tensor<Rank3<1, IMAGE_SIZE, IMAGE_SIZE>, f32, _> = device.tensor(image);
    let output = NETWORK.forward(input);

    output.as_vec().iter().map(|n| *n as f64).collect()
}
