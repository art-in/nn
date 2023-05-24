use super::{dimensions::Dimensions, pos::VirtualPosition};

pub struct Rect {
    pos: VirtualPosition,
    dim: Dimensions,
}

impl Rect {
    pub fn new(pos: VirtualPosition, dim: Dimensions) -> Self {
        Self::assert_bounds(&pos, &dim);
        Self { pos, dim }
    }

    pub fn new_unchecked(pos: VirtualPosition, dim: Dimensions) -> Self {
        Self { pos, dim }
    }

    pub fn move_to(&mut self, new_pos: VirtualPosition) {
        Self::assert_bounds(&new_pos, &self.dim);
        self.pos = new_pos;
    }

    pub fn pos(&self) -> &VirtualPosition {
        &self.pos
    }

    pub fn dim(&self) -> &Dimensions {
        &self.dim
    }

    fn assert_bounds(pos: &VirtualPosition, dim: &Dimensions) {
        assert!(pos.x() + dim.width() <= 1.0, "out of horizontal bounds");
        assert!(pos.y() + dim.height() <= 1.0, "out of vertical bounds");
    }
}
