use crate::primitives::{dimensions::Dimensions, pos::VirtualPosition, rect::Rect};

pub struct SceneBounds {
    top_left_right: Vec<Rect>,
    bottom: Rect,
}

impl SceneBounds {
    pub fn new() -> Self {
        let top = Rect::new_unchecked(
            VirtualPosition::new_unchecked(0.0, -1.0),
            Dimensions::new(1.0, 1.0),
        );

        let left = Rect::new_unchecked(
            VirtualPosition::new_unchecked(-1.0, 0.0),
            Dimensions::new(1.0, 1.0),
        );

        let right = Rect::new_unchecked(
            VirtualPosition::new_unchecked(1.0, 0.0),
            Dimensions::new(1.0, 1.0),
        );

        let bottom = Rect::new_unchecked(
            VirtualPosition::new_unchecked(0.0, 1.0),
            Dimensions::new(1.0, 1.0),
        );

        Self {
            top_left_right: vec![top, left, right],
            bottom,
        }
    }

    pub fn top_left_right(&self) -> &Vec<Rect> {
        &self.top_left_right
    }

    pub fn bottom(&self) -> &Rect {
        &self.bottom
    }
}

impl Default for SceneBounds {
    fn default() -> Self {
        Self::new()
    }
}
