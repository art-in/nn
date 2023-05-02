use std::time::Duration;

use autograd::{pool::BValPool, val::BVal};
use criterion::{criterion_group, criterion_main, Criterion};

#[inline]
fn val_forward() -> BVal {
    let pool = BValPool::default();

    let x1 = pool.pull(1.0);
    let x2 = pool.pull(2.0);

    // all possible math operations
    let mul = &x1 * &x2;
    let add = &mul + &x1;
    let div = &add / &x2;
    let neg = -&div;
    let pow = x2.pow_val(&neg);
    let sub = &pow - &neg;
    let tanh = sub.tanh();

    assert_eq!(tanh.borrow().d, 0.9520794848173941);

    tanh
}

pub fn val_forward_benchmark(c: &mut Criterion) {
    c.bench_function("val_forward", |b| b.iter(|| val_forward()));
}

pub fn val_backward_benchmark(c: &mut Criterion) {
    let val = val_forward();
    c.bench_function("val_backward", |b| b.iter(|| val.backward()));
}

criterion_group! {
    name = val_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(5000)
        .noise_threshold(0.05);
    targets = val_forward_benchmark, val_backward_benchmark
}
criterion_main!(val_benches);
