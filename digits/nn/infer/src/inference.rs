use crate::model::NETWORK;

// image represented as series of pixels, where each each pixel is a number in range [-1, 1]
pub fn infer(image: &[f64]) -> Vec<f64> {
    let output = NETWORK.with(|n| n.forward(image));
    output.iter().map(|v| v.borrow().d).collect()
}
