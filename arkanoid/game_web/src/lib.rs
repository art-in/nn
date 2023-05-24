use game::GameWebRc;
use wasm_bindgen::prelude::*;

mod drawer;
mod game;
mod utils;

#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(start)]
pub fn main_js() {
    #[cfg(debug_assertions)]
    console_error_panic_hook::set_once();

    GameWebRc::start();
}
