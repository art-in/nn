// required for dxdf
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::fs::metadata;

use dfdx::{
    prelude::{DeviceBuildExt, LoadFromNpz},
    tensor::AutoDevice,
};
use model_type::Model;

mod game_iterator;
mod image_drawer;
mod model_type;
mod test;
mod train;
mod utils;

pub const MODEL_PATH: &str = "./models/arkanoid.npz";

fn main() {
    let device = AutoDevice::default();
    let mut model = device.build_module::<Model, f32>();

    if metadata(MODEL_PATH).is_ok() {
        println!("loading model from file: {MODEL_PATH}");
        model.load(MODEL_PATH).expect("failed to load model");
    }

    train::train(&device, &mut model);
    test::test(&device, &model, true);
}
