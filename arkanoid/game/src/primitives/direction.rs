use std::f64::consts::PI;

// direction in radians, counter-clockwise from the right (e.g. 0 is right, PI/2 - is top)
pub struct Direction(f64);

impl Direction {
    pub fn new(d: f64) -> Self {
        let normalized = d % (2.0 * PI);
        Self(normalized)
    }

    pub fn angle(&self) -> f64 {
        self.0
    }
}
