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
        vec![784, 200, 80, 10],
        MODELS_DIR,
        MODEL_FILE_NAME_PREFIX,
    );

    training::train(&mut net);

    net.serialize_to_file(MODELS_DIR, MODEL_FILE_NAME_PREFIX);

    testing::test(&net);
}
