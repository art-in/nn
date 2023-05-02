use std::time::Duration;

use autograd::network::Network;
use criterion::{criterion_group, criterion_main, Criterion};

#[inline]
fn classification() {
    let inputs: Vec<Vec<f64>> = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];
    let expecteds: Vec<f64> = vec![1.0, -1.0, -1.0, 1.0];

    let net = Network::new(vec![3, 4, 4, 1]);

    let mut last_total_loss = 0.0;

    for _ in 0..100 {
        // forward
        let mut total_loss = net.pool.pull(0.0);
        for (input, expected) in inputs.iter().zip(expecteds.iter()) {
            let output = &net.forward(input)[0];
            let loss = (*expected - output).pow(2.0);
            total_loss = &total_loss + &loss;
            last_total_loss = total_loss.borrow().d;
        }

        // backward
        net.reset_grad();

        total_loss.borrow_mut().grad = 1.0;
        total_loss.backward();

        // update
        for param in &net.parameters() {
            let grad = param.borrow().grad;
            param.borrow_mut().d -= 0.05 * grad;
        }
    }

    assert!(last_total_loss < 0.1);
}

pub fn classification_benchmark(c: &mut Criterion) {
    c.bench_function("classification_benchmark", |b| b.iter(|| classification()));
}

criterion_group! {
    name = network_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(30))
        .sample_size(2500)
        .noise_threshold(0.05);
    targets = classification_benchmark
}
criterion_main!(network_benches);
