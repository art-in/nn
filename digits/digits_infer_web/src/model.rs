use once_cell::sync::Lazy;
use wasm_bindgen::JsValue;
use web_sys::console;

use network::network::Network;

// large models fail in dev build with stack overflow error while dropping lots
// of termporary BVals which are produced by network forward pass, so use small
// model for dev build and switch to large model in release mode only
#[cfg(not(debug_assertions))]
static MODEL: &[u8] = include_bytes!("../models/digits-784-1000-400-150-10.nm");

#[cfg(debug_assertions)]
static MODEL: &[u8] = include_bytes!("../models/digits-784-30-10.nm");

thread_local!(pub static NETWORK: Lazy<Network> = Lazy::new(
    || Network::deserialize_from_reader(MODEL))
);

pub fn init_model() {
    // force model init earlier, so it doesn't slow down real first use
    console::warn_1(&JsValue::from("initializing network"));
    NETWORK.with(|net| {
        console::log_1(&JsValue::from(format!(
            "network parameters count: {}",
            net.parameters().len()
        )));
    })
}
