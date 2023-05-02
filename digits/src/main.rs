use autograd::network::Network;

mod image;
pub mod mnist;
mod testing;
pub mod training;
mod utils;

const MODELS_DIR: &str = "digits/models";
const MODEL_FILE_NAME_PREFIX: &str = "digits";

fn main() {
    let mut net = Network::new_or_deserialize_from_file(
        vec![784, 800, 300, 10],
        MODELS_DIR,
        MODEL_FILE_NAME_PREFIX,
    );

    training::train(&mut net, MODELS_DIR, MODEL_FILE_NAME_PREFIX);

    testing::test(&net, false);
}
