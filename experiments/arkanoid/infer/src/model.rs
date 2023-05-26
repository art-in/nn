use std::io::Cursor;

use dfdx::prelude::*;
use dfdx::tensor::AutoDevice;
use once_cell::sync::Lazy;
use zip::ZipArchive;

use crate::model_type::{Model, ModelBuild};

static MODEL: &[u8] = include_bytes!("../../train/models/arkanoid.npz");

pub static NETWORK: Lazy<ModelBuild> = Lazy::new(|| {
    let device = AutoDevice::default();
    let mut model = device.build_module::<Model, f32>();

    let reader = Cursor::new(MODEL);
    let mut zip = ZipArchive::new(reader).expect("failed to read model archive");
    model.read(&mut zip).expect("failed to read model");

    model
});
