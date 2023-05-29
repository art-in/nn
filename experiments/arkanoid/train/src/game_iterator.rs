use arkanoid_game::{game::Game, state::draw::DrawGameState};
use rand::Rng;

use crate::{image_drawer::ImageDrawer, utils::onehot};

pub const PX_SIZE: usize = 4;
pub const IMAGE_PX_SIZE: usize = 100;
pub const IMAGE_SIZE: usize = IMAGE_PX_SIZE * IMAGE_PX_SIZE * PX_SIZE;
pub const PREDICTION_POSITIONS: usize = 10;

const DUMP_IMAGES: bool = false;

pub struct GameIterator {
    max_steps: usize,
    game: Game<ImageDrawer>,
    step_idx: u32,
}

impl GameIterator {
    pub fn new(max_steps: usize) -> Self {
        let mut game = Game::new(ImageDrawer::new(IMAGE_PX_SIZE as u32, IMAGE_PX_SIZE as u32));
        game.set_stop_flags(false, false);

        Self {
            max_steps,
            game,
            step_idx: 0,
        }
    }
}

impl Iterator for GameIterator {
    type Item = (Vec<f32>, Vec<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.step_idx >= self.max_steps as u32 {
            None
        } else {
            self.game.step();

            let mut rng = rand::thread_rng();
            self.game.state_mut().platform_mut().set_pos_x(
                map_prediction_position_to_game_position(
                    rng.gen_range(0..PREDICTION_POSITIONS as u32),
                    PREDICTION_POSITIONS,
                ),
            );

            self.game.draw();

            let image = self.game.drawer().get_image();
            let label = get_platform_optimal_prediction_position(&self.game, PREDICTION_POSITIONS);

            if DUMP_IMAGES {
                image
                    .save(format!("images/{}-{}.png", self.step_idx, label))
                    .expect("failed to save image");
            }

            let image: Vec<f32> = image.as_raw().iter().map(|n| *n as f32 / 255.0).collect();
            let label = onehot(PREDICTION_POSITIONS, label);

            assert_eq!(image.len(), IMAGE_SIZE, "wrong image size");

            self.step_idx += 1;

            Some((image, label))
        }
    }
}

impl ExactSizeIterator for GameIterator {
    fn len(&self) -> usize {
        self.max_steps
    }
}

fn map_game_position_to_prediction_position(x: f32, positions_count: usize) -> u32 {
    let step = 1.0 / positions_count as f32;
    (x / step) as u32
}

pub fn map_prediction_position_to_game_position(pos: u32, positions_count: usize) -> f64 {
    let step = 1.0 / positions_count as f64;
    pos as f64 * step
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
