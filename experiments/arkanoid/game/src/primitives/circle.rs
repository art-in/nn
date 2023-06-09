use super::pos::VirtualPosition;

pub struct Circle {
    pos: VirtualPosition,
    radius: f64,
}

impl Circle {
    pub fn new(pos: VirtualPosition, radius: f64) -> Self {
        Self::assert_bounds(&pos, radius);
        Self { pos, radius }
    }

    pub fn new_unchecked(pos: VirtualPosition, radius: f64) -> Self {
        Self { pos, radius }
    }

    pub fn set_pos(&mut self, new_pos: VirtualPosition) {
        Self::assert_bounds(&new_pos, self.radius);
        self.pos = new_pos;
    }

    pub fn set_pos_clamped(&mut self, new_pos: VirtualPosition) {
        let new_pos_clamped = VirtualPosition::new(
            new_pos.x().max(0.0 + self.radius).min(1.0 - self.radius),
            new_pos.y().max(0.0 + self.radius).min(1.0 - self.radius),
        );
        self.set_pos(new_pos_clamped);
    }

    pub fn pos(&self) -> &VirtualPosition {
        &self.pos
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }

    fn assert_bounds(pos: &VirtualPosition, radius: f64) {
        assert!(
            pos.x() >= radius,
            "out of horizontal bounds, x: {}",
            pos.x() - radius
        );
        assert!(
            pos.y() >= radius,
            "out of vertical bounds, y: {}",
            pos.y() - radius
        );

        assert!(
            pos.x() + radius <= 1.0,
            "out of horizontal bounds, x: {}",
            pos.x() + radius
        );
        assert!(
            pos.y() + radius <= 1.0,
            "out of vertical bounds, y: {}",
            pos.y() + radius
        );
    }
}
