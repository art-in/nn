use std::io::Cursor;

use dfdx::prelude::*;
use dfdx::tensor::AutoDevice;
use once_cell::sync::Lazy;
use zip::ZipArchive;

use crate::model_type::{Model, ModelBuild};

static MODEL: &[u8] = include_bytes!(
    "../../train/models/conv-784-32C3-32C3-32C5S2P2-64C3-64C3-64C5S2P2-128-10-sets-2/mnist.npz"
);

pub static NETWORK: Lazy<ModelBuild> = Lazy::new(|| {
    let device = AutoDevice::default();
    let mut model = device.build_module::<Model, f32>();

    let reader = Cursor::new(MODEL);
    let mut zip = ZipArchive::new(reader).expect("failed to read model archive");
    model.read(&mut zip).expect("failed to read model");

    model
});

pub fn init_model() -> usize {
    // force model init earlier, so it doesn't slow down real first use
    NETWORK.num_trainable_params()
}
