use wasm_bindgen::JsValue;
use web_sys::console;

#[allow(dead_code)]
pub fn console_log(s: String) {
    console::log_1(&JsValue::from(s));
}
