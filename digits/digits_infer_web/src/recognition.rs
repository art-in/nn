use wasm_bindgen::prelude::*;

use crate::model::NETWORK;

const IMAGE_SIZE: usize = 28;

#[wasm_bindgen]
pub fn recognize(image: &[f64]) -> Vec<f64> {
    assert_eq!(image.len(), IMAGE_SIZE.pow(2), "invalid image size");

    let output = NETWORK.with(|n| n.forward(image));
    output.iter().map(|v| v.borrow().d).collect()
}
