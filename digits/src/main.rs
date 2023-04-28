use std::fs;

use autograd::network::Network;

mod image;
pub mod mnist;
mod testing;
pub mod training;
mod utils;

fn main() {
    const MODEL_PATH: &str = "digits/models/digits-784-100-10.nm";

    let mut net = if fs::metadata(MODEL_PATH).is_ok() {
        Network::deserialize_from_file(MODEL_PATH)
    } else {
        Network::new(vec![784, 100, 10])
    };

    training::train(&mut net);

    net.serialize_to_file(&format!(
        "digits/models/{}",
        &utils::get_network_model_file_name("digits", &net)
    ));

    testing::test(&net);
}
