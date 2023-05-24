use wasm_bindgen::JsValue;

use super::size::Size;

pub fn resize_canvas(
    canvas: &web_sys::HtmlCanvasElement,
    target_css_size: Size,
    pixel_ratio: f64,
) -> Result<(), JsValue> {
    canvas.set_width((target_css_size.width * pixel_ratio) as u32);
    canvas.set_height((target_css_size.height * pixel_ratio) as u32);

    let style = &canvas.style();
    style.set_property("width", &format!("{}px", target_css_size.width))?;
    style.set_property("height", &format!("{}px", target_css_size.height))?;

    Ok(())
}
