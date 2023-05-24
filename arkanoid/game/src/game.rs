use super::{
    geometry::{collision::collide_circle_and_rect, reflection::reflect_angle_by_normal},
    state::{draw::DrawGameState, State},
};
use crate::primitives::direction::Direction;

pub struct Game<TDrawer: DrawGameState> {
    state: State,
    drawer: TDrawer,
    stop_on_fail: bool,
    stop_on_win: bool,
}

#[derive(PartialEq, Eq)]
pub enum GameStatus {
    Failed,
    Won,
    InProgress,
}

impl<TDrawer: DrawGameState> Game<TDrawer> {
    pub fn new(drawer: TDrawer) -> Self {
        Self {
            state: State::new(),
            drawer,
            stop_on_fail: true,
            stop_on_win: true,
        }
    }

    pub fn draw(&self) {
        self.drawer.draw(&self.state);
    }

    pub fn set_stop_flags(&mut self, stop_on_fail: bool, stop_on_win: bool) {
        self.stop_on_fail = stop_on_fail;
        self.stop_on_win = stop_on_win;
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    pub fn step(&mut self) -> GameStatus {
        let new_ball_bounds = self.state.ball().step();

        let mut collision: Option<_> = None;

        // check if ball collides with scene bounds
        if collision.is_none() {
            for bounds in self.state.scene_bounds().top_left_right() {
                let col = collide_circle_and_rect(&new_ball_bounds, bounds);

                if col.is_some() {
                    collision = col;
                    break;
                }
            }

            let col = collide_circle_and_rect(&new_ball_bounds, self.state.scene_bounds().bottom());

            if col.is_some() {
                collision = col;

                if self.stop_on_fail {
                    return GameStatus::Failed;
                }
            }
        }

        // check if ball collides with blocks
        if collision.is_none() {
            let mut is_block_deactivated = false;
            for block in self.state.block_set_mut().active_blocks_mut() {
                let col = collide_circle_and_rect(&new_ball_bounds, block.bounds());

                if col.is_some() {
                    block.set_is_active(false);
                    is_block_deactivated = true;
                    collision = col;
                    break;
                }
            }

            if is_block_deactivated {
                self.state.ball_mut().accel();
            }

            if !self.state.block_set().has_active_blocks() && self.stop_on_win {
                return GameStatus::Won;
            }
        }

        // check if ball collides with the platform
        if collision.is_none() {
            let col = collide_circle_and_rect(&new_ball_bounds, self.state.platform().bounds());

            if col.is_some() {
                collision = col;
            }
        }

        if let Some(collision) = collision {
            // bounce ball out of collision point
            self.state.ball_mut().set_pos(collision.circle_center);
            let reflected_angle =
                reflect_angle_by_normal(self.state.ball().dir().angle(), collision.rect_norm);
            self.state
                .ball_mut()
                .set_dir(Direction::new(reflected_angle));
        } else {
            self.state.ball_mut().set_pos(new_ball_bounds.pos().clone());
        }

        GameStatus::InProgress
    }

    pub fn move_platform_to(&mut self, center_virtual_x: f64) {
        self.state.platform_mut().move_to(center_virtual_x);
    }
}
