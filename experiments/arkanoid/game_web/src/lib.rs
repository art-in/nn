use game::GameWebRc;
use wasm_bindgen::prelude::*;

mod canvas_drawer;
mod game;
mod utils;

#[wasm_bindgen(start)]
pub fn main_js() {
    #[cfg(debug_assertions)]
    console_error_panic_hook::set_once();

    GameWebRc::start();
}
