use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use digits_train::train::train;
use network::network::Network;

const TRAIN_IMAGES_FILE_PATH: &str = "../digits_train/data/train-images-idx3-ubyte";
const TRAIN_LABELS_FILE_PATH: &str = "../digits_train/data/train-labels-idx1-ubyte";

pub fn train_benchmark(c: &mut Criterion) {
    let mut net = Network::new(vec![784, 200, 80, 10]);

    c.bench_function("train-784-200-80-10", |b| {
        b.iter(|| {
            train(
                &mut net,
                TRAIN_IMAGES_FILE_PATH,
                TRAIN_LABELS_FILE_PATH,
                "/tmp",
                "digits",
                "/tmp",
                1,
                10,
                10,
                (0.01, 0.01),
                Some(10),
                Some(10),
            )
        })
    });
}

criterion_group! {
    name = train_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(10))
        .measurement_time(Duration::from_secs(60))
        .sample_size(10)
        .noise_threshold(0.05);
    targets = train_benchmark
}
criterion_main!(train_benches);
