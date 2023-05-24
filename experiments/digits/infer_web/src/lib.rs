use dfdx_infer::{inference::infer, model::init_model};
use wasm_bindgen::prelude::*;
use web_sys::console;

#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(start)]
pub fn main_js() -> Result<(), JsValue> {
    #[cfg(debug_assertions)]
    console_error_panic_hook::set_once();
    init();
    Ok(())
}

pub fn init() {
    console::warn_1(&JsValue::from("initializing network"));
    console::log_1(&JsValue::from(format!(
        "network parameters count: {}",
        init_model()
    )));
}

const IMAGE_SIZE: usize = 28;

#[wasm_bindgen]
pub fn infer_digit(image: &[f64]) -> Vec<f64> {
    assert_eq!(image.len(), IMAGE_SIZE.pow(2), "invalid image size");
    infer(image)
}
