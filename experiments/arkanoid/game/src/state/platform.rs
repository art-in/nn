use crate::primitives::{dimensions::Dimensions, pos::VirtualPosition, rect::Rect};

const PLATFORM_HEIGHT: f64 = 0.05;
const PLATFORM_WIDTH: f64 = 0.3;

pub struct Platform {
    bounds: Rect,
}

impl Platform {
    pub fn new() -> Self {
        let pos = VirtualPosition::new(0.0, 1.0 - PLATFORM_HEIGHT);
        let bounds = Rect::new(pos, Dimensions::new(PLATFORM_WIDTH, PLATFORM_HEIGHT));

        Self { bounds }
    }

    pub fn bounds(&self) -> &Rect {
        &self.bounds
    }

    pub fn set_pos_x(&mut self, virtual_x: f64) {
        let virtual_x = virtual_x.max(0.0).min(1.0 - PLATFORM_WIDTH);
        self.bounds
            .move_to(VirtualPosition::new(virtual_x, self.bounds.pos().y()));
    }

    pub fn set_pos_x_center(&mut self, center_virtual_x: f64) {
        let x = center_virtual_x - PLATFORM_WIDTH / 2.0;
        self.set_pos_x(x);
    }
}

impl Default for Platform {
    fn default() -> Self {
        Self::new()
    }
}
