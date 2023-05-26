use std::{collections::VecDeque, f64::consts::PI, time::Duration};

use arkanoid_game::{
    game::Game,
    primitives::{circle::Circle, direction::Direction, pos::VirtualPosition, rect::Rect},
    state::draw::DrawGameState,
};
use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;

struct FakeDrawer {}
impl DrawGameState for FakeDrawer {
    fn clear(&mut self) {}
    fn rect(&mut self, _: &Rect, _: &str, _: &str) {}
    fn circle(&mut self, _: &Circle, _: &str, _: &str) {}
}

#[inline]
fn play_game() {
    let mut game = Game::new(FakeDrawer {});
    game.set_stop_flags(false, false);

    let mut rng = rand::thread_rng();

    game.state_mut()
        .ball_mut()
        .set_pos(VirtualPosition::new(0.5, 0.5))
        .set_dir(Direction::new(rng.gen_range(-PI..PI)));

    let mut ball_prev_positions = VecDeque::new();

    let mut step_idx = 0;
    const MAX_STEPS: i32 = 10_000;

    while step_idx < MAX_STEPS {
        game.step();

        // also make sure ball doesn't stuck in one position, which may happen when ball direction
        // is very close to be perpendicular with collision normal (e.g. when ball slightly touches
        // rectangle corner), in this case direction change is too small and next step collides again.
        // moving ball out of touch immediately after collision solves the problem, this test just
        // makes sure ball doesn't stuck in one place or in close loop for some other reason
        ball_prev_positions.push_back(game.state().ball().bounds().pos().clone());

        if ball_prev_positions.len() > 100 {
            ball_prev_positions.pop_front();
            let ball_movement_area = get_ball_movement_area(&ball_prev_positions);
            assert!(
                ball_movement_area >= 0.2,
                "ball stuck on step {step_idx} in area {ball_movement_area}"
            );
        }

        step_idx += 1;
    }
}

fn get_ball_movement_area(positions: &VecDeque<VirtualPosition>) -> f64 {
    let mut range_x: (f64, f64) = (1.0, 0.0);
    let mut range_y: (f64, f64) = (1.0, 0.0);

    for pos in positions.iter() {
        range_x.0 = range_x.0.min(pos.x());
        range_x.1 = range_x.1.max(pos.x());

        range_y.0 = range_y.0.min(pos.y());
        range_y.1 = range_y.1.max(pos.y());
    }

    let diff_x = range_x.1 - range_x.0;
    let diff_y = range_y.1 - range_y.0;

    (diff_x.powf(2.0) + diff_y.powf(2.0)).sqrt()
}

pub fn arkanoid_benchmark(c: &mut Criterion) {
    c.bench_function("arkanoid_benchmark", |b| b.iter(|| play_game()));
}

criterion_group! {
    name = arkanoid_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(60))
        .sample_size(2500)
        .noise_threshold(0.05);
    targets = arkanoid_benchmark
}
criterion_main!(arkanoid_benches);
