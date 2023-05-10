// required for dxdf
#![feature(generic_const_exprs)]

use std::fs::metadata;

use dfdx::{
    prelude::{DeviceBuildExt, LoadFromNpz},
    tensor::AutoDevice,
};
use model_type::Model;

mod model_type;
mod test;
mod train;

fn main() {
    let dev = AutoDevice::default();
    let mut model = dev.build_module::<Model, f32>();

    if metadata("./models/mnist.npz").is_ok() {
        println!("loading model from file");
        model
            .load("./models/mnist.npz")
            .expect("failed to load model");
    }

    train::train(&dev, &mut model);
}
