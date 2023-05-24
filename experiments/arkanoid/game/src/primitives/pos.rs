// position in a world with dimensions 1x1, where top-left is (0;0), bottom-right is (1;1).
// this allows us to map to drawing surface of any "real" pixel dimensions
#[derive(Clone)]
pub struct VirtualPosition {
    x: f64,
    y: f64,
}

impl VirtualPosition {
    pub fn new(x: f64, y: f64) -> Self {
        assert!((0.0..=1.0).contains(&x), "invalid position x: {x:.2}");
        assert!((0.0..=1.0).contains(&y), "invalid position y: {y:.2}");

        Self { x, y }
    }

    pub fn new_unchecked(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn x(&self) -> f64 {
        self.x
    }

    pub fn y(&self) -> f64 {
        self.y
    }
}
