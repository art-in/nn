use dump::dump_dataset_images;
use mnistify::mnistify;

mod dump;
mod mnistify;
mod padding;
mod utils;

pub const MNIST_PATH: &str = "../data/mnist";

const TRAIN_PATH: &str = "../data/canvas_digits/sources/train";
const TEST_PATH: &str = "../data/canvas_digits/sources/test";
const OUTPUT_PATH: &str = "../data/canvas_digits";

fn main() {
    mnistify(TRAIN_PATH, TEST_PATH, OUTPUT_PATH);
    dump_dataset_images(OUTPUT_PATH);
}
