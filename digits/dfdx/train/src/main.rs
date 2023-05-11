// required for dxdf
#![feature(generic_const_exprs)]

use std::fs::metadata;

use dfdx::{
    prelude::{DeviceBuildExt, LoadFromNpz},
    tensor::AutoDevice,
};
use model_type::Model;

mod data;
mod data_aug;
mod model_type;
mod test;
mod train;
mod utils;

const MODEL_PATH: &str = "./models/mnist.npz";

fn main() {
    let device = AutoDevice::default();
    let mut model = device.build_module::<Model, f32>();

    if metadata(MODEL_PATH).is_ok() {
        println!("loading model from file: {MODEL_PATH}");
        model.load(MODEL_PATH).expect("failed to load model");
    }

    train::train(&device, &mut model, MODEL_PATH);

    test::test(&model, true);
}
