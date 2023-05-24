use std::f64::consts::PI;

use arkanoid_game::{
    primitives::{circle::Circle, rect::Rect},
    state::draw::DrawGameState,
};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

pub struct CanvasGameStateDrawer {
    canvas: HtmlCanvasElement,
    ctx: CanvasRenderingContext2d,
}

impl CanvasGameStateDrawer {
    pub fn new(canvas: HtmlCanvasElement) -> Self {
        let ctx = canvas
            .get_context("2d")
            .unwrap()
            .expect("failed to get 2D context from canvas")
            .dyn_into::<CanvasRenderingContext2d>()
            .unwrap();

        Self { ctx, canvas }
    }
}

impl DrawGameState for CanvasGameStateDrawer {
    fn clear(&self) {
        let canvas_width = self.canvas.width() as f64;
        let canvas_height = self.canvas.height() as f64;

        self.ctx.clear_rect(0.0, 0.0, canvas_width, canvas_height);
    }

    fn rect(&self, bounds: &Rect, color: &str, border_color: &str) {
        let canvas_width = self.canvas.width() as f64;
        let canvas_height = self.canvas.height() as f64;

        let x = bounds.pos().x() * canvas_width;
        let y = bounds.pos().y() * canvas_height;
        let width = bounds.dim().width() * canvas_width;
        let height = bounds.dim().height() * canvas_height;

        self.ctx.set_fill_style(&JsValue::from(color));
        self.ctx.fill_rect(x, y, width, height);

        self.ctx.set_stroke_style(&JsValue::from(border_color));
        self.ctx.stroke_rect(x, y, width, height);
    }

    fn circle(&self, bounds: &Circle, color: &str, border_color: &str) {
        let canvas_width = self.canvas.width() as f64;
        let canvas_height = self.canvas.height() as f64;

        let x = bounds.pos().x() * canvas_width;
        let y = bounds.pos().y() * canvas_height;

        let radius_x = bounds.radius() * canvas_width;
        let radius_y = bounds.radius() * canvas_height;

        let start_angle = 0.0;
        let end_angle = 2.0 * PI;

        self.ctx.set_fill_style(&JsValue::from(color));
        self.ctx.set_stroke_style(&JsValue::from(border_color));

        self.ctx.begin_path();
        self.ctx
            .ellipse(x, y, radius_x, radius_y, 0.0, start_angle, end_angle)
            .expect("failed to draw a circle");
        self.ctx.fill();
        self.ctx.stroke();
    }
}
