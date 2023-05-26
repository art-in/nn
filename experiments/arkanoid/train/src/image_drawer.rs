use arkanoid_game::{
    primitives::{circle::Circle, dimensions::Dimensions, pos::VirtualPosition, rect::Rect},
    state::draw::DrawGameState,
};
use colors_transform::Color;
use image::{ImageBuffer, Rgba, RgbaImage};

pub struct ImageDrawer {
    width: u32,
    height: u32,
    image: RgbaImage,
}

impl ImageDrawer {
    pub fn new(width: u32, height: u32) -> Self {
        let image = ImageBuffer::new(width, height);

        Self {
            width,
            height,
            image,
        }
    }

    pub fn get_image(&self) -> &RgbaImage {
        &self.image
    }

    fn fill_rect(&mut self, bounds: &Rect, color: &str, is_transparent: bool) {
        let color = colors_transform::Rgb::from_hex_str(color).expect("failed to parse color");
        let color = [
            color.get_red() as u8,
            color.get_green() as u8,
            color.get_blue() as u8,
            if is_transparent { 0 } else { 255 },
        ];

        let rect_x = (bounds.pos().x() * self.width as f64) as u32;
        let rect_y = (bounds.pos().y() * self.height as f64) as u32;

        let rect_width = (bounds.dim().width() * self.width as f64) as u32;
        let rect_height = (bounds.dim().height() * self.height as f64) as u32;

        for x in rect_x..(rect_x + rect_width) {
            for y in rect_y..(rect_y + rect_height) {
                self.image.put_pixel(x, y, Rgba(color));
            }
        }
    }
}

impl DrawGameState for ImageDrawer {
    fn clear(&mut self) {
        self.fill_rect(
            &Rect::new(VirtualPosition::new(0.0, 0.0), Dimensions::new(1.0, 1.0)),
            "#000000",
            true,
        );
    }

    fn rect(&mut self, bounds: &Rect, color: &str, _border_color: &str) {
        self.fill_rect(bounds, color, false);
    }

    fn circle(&mut self, bounds: &Circle, color: &str, _border_color: &str) {
        self.fill_rect(
            &Rect::new(
                VirtualPosition::new(
                    bounds.pos().x() - bounds.radius(),
                    bounds.pos().y() - bounds.radius(),
                ),
                Dimensions::new(bounds.radius() * 2.0, bounds.radius() * 2.0),
            ),
            color,
            false,
        );
    }
}
