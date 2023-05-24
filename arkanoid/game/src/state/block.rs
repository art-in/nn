use crate::primitives::rect::Rect;

pub struct Block {
    bounds: Rect,
    is_active: bool,
}

impl Block {
    pub fn new(bounds: Rect) -> Self {
        Self {
            bounds,
            is_active: true,
        }
    }

    pub fn bounds(&self) -> &Rect {
        &self.bounds
    }

    pub fn is_active(&self) -> bool {
        self.is_active
    }

    pub fn set_is_active(&mut self, val: bool) {
        self.is_active = val;
    }
}
