use digits_train::{test::test, train::train};
use network::network::Network;

const TRAIN_IMAGES_FILE_PATH: &str = "digits/digits_train/data/train-images-idx3-ubyte";
const TRAIN_LABELS_FILE_PATH: &str = "digits/digits_train/data/train-labels-idx1-ubyte";
const TEST_IMAGES_FILE_PATH: &str = "digits/digits_train/data/t10k-images-idx3-ubyte";
const TEST_LABELS_FILE_PATH: &str = "digits/digits_train/data/t10k-labels-idx1-ubyte";

const MODEL_FILE_NAME_PREFIX: &str = "digits";
const MODELS_DIR: &str = "digits/digits_train_runner/models";
const PLOTS_DIR: &str = "digits/digits_train_runner/plots";
const FAILED_IMAGES_DIR: &str = "digits/digits_train_runner/images";

const EPOCHS: u32 = 1;
const BATCHES: u32 = 6000;
const BATCH_SIZE: u32 = 10;
const LEARNING_RATE: (f64, f64) = (0.01, 0.01);
const PLOT_LOSSES_EACH_NTH_BATCH: u32 = 50;
const SERIALIZE_MODEL_EACH_NTH_BATCH: u32 = 100;

fn main() {
    let mut net = Network::new_or_deserialize_from_file(
        vec![784, 200, 80, 10],
        MODELS_DIR,
        MODEL_FILE_NAME_PREFIX,
    );

    train(
        &mut net,
        TRAIN_IMAGES_FILE_PATH,
        TRAIN_LABELS_FILE_PATH,
        MODELS_DIR,
        MODEL_FILE_NAME_PREFIX,
        PLOTS_DIR,
        EPOCHS,
        BATCHES,
        BATCH_SIZE,
        LEARNING_RATE,
        Some(PLOT_LOSSES_EACH_NTH_BATCH),
        Some(SERIALIZE_MODEL_EACH_NTH_BATCH),
    );

    test(
        &net,
        TEST_IMAGES_FILE_PATH,
        TEST_LABELS_FILE_PATH,
        FAILED_IMAGES_DIR,
        false,
    );
}
