pub struct Dimensions {
    width: f64,
    height: f64,
}

impl Dimensions {
    pub fn new(width: f64, height: f64) -> Self {
        assert!(width > 0.0 && width <= 1.0, "invalid width");
        assert!(height > 0.0 && height <= 1.0, "invalid height");
        Self { width, height }
    }

    pub fn width(&self) -> f64 {
        self.width
    }

    pub fn height(&self) -> f64 {
        self.height
    }
}
