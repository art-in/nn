use network::network::Network;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn main_js() -> Result<(), JsValue> {
    #[cfg(debug_assertions)]
    console_error_panic_hook::set_once();
    Ok(())
}

#[wasm_bindgen]
pub fn say() -> String {
    let network = Network::new(vec![100, 10, 1]);
    let res = network.layers[0].neurons[0].bias.borrow().d.to_string();
    res
}
