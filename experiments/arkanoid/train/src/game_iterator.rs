use std::cell::Cell;

use arkanoid_game::{game::Game, primitives::pos::VirtualPosition, state::draw::DrawGameState};
use dfdx::data::ExactSizeDataset;

use crate::{image_drawer::ImageDrawer, utils::onehot};

pub const PX_SIZE: usize = 4;
pub const IMAGE_PX_SIZE: usize = 100;
pub const IMAGE_SIZE: usize = IMAGE_PX_SIZE * IMAGE_PX_SIZE * PX_SIZE;
pub const PREDICTION_POSITIONS: usize = 10;

const DUMP_IMAGES: bool = false;

pub struct GameIterator {
    max_steps: usize,
    get_idx: Cell<u32>,
}

impl GameIterator {
    pub fn new(max_steps: usize) -> Self {
        Self {
            max_steps,
            get_idx: Cell::new(0),
        }
    }
}

impl ExactSizeDataset for GameIterator {
    type Item<'a> = (Vec<f32>, Vec<f32>) where Self: 'a;

    fn get(&self, step_idx: usize) -> Self::Item<'_> {
        let steps_per_row_and_col: f64 = (self.max_steps as f64).sqrt();
        let step_distance = 1.0 / steps_per_row_and_col;

        let step_x = step_idx as f64 % steps_per_row_and_col;
        let step_y = step_idx as f64 / steps_per_row_and_col;

        let ball_x = step_x * step_distance;
        let ball_y = step_y * step_distance;

        let mut game = Game::new(ImageDrawer::new(IMAGE_PX_SIZE as u32, IMAGE_PX_SIZE as u32));

        game.state_mut()
            .ball_mut()
            .bounds_mut()
            .set_pos_clamped(VirtualPosition::new(ball_x, ball_y));

        game.draw();

        let image = game.drawer().get_image();
        let label = get_platform_optimal_prediction_position(&game, PREDICTION_POSITIONS);

        if DUMP_IMAGES {
            image
                .save(format!("images/{}-{}.png", self.get_idx.get(), label))
                .expect("failed to save image");
        }

        let image: Vec<f32> = image.as_raw().iter().map(|n| *n as f32 / 255.0).collect();
        let label = onehot(PREDICTION_POSITIONS, label);

        assert_eq!(image.len(), IMAGE_SIZE, "wrong image size");

        self.get_idx.set(self.get_idx.get() + 1);

        (image, label)
    }

    fn len(&self) -> usize {
        self.max_steps
    }
}

fn map_game_position_to_prediction_position(x: f32, positions_count: usize) -> u32 {
    let step = 1.0 / positions_count as f32;
    (x / step) as u32
}

fn get_platform_optimal_prediction_position(
    game: &Game<impl DrawGameState>,
    positions_count: usize,
) -> u32 {
    map_game_position_to_prediction_position(
        game.state().ball().bounds().pos().x() as f32,
        positions_count,
    )
}
