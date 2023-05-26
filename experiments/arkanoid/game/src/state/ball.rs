use std::f64::consts::PI;

use rand::Rng;

use crate::primitives::{circle::Circle, direction::Direction, pos::VirtualPosition};

const BALL_RADIUS: f64 = 0.03;
const BALL_POS_INITIAL: (f64, f64) = (0.5, 0.5);
const BALL_VELOCITY_INITIAL: f64 = 0.01;
const BALL_VELOCITY_MAX: f64 = 0.015;
const BALL_ACCELL_RATIO: f64 = 1.05;

pub struct Ball {
    bounds: Circle,
    dir: Direction,
    velocity: f64,
}

impl Ball {
    pub fn new() -> Self {
        let bounds = Circle::new(
            VirtualPosition::new(BALL_POS_INITIAL.0, BALL_POS_INITIAL.1),
            BALL_RADIUS,
        );

        let mut rng = rand::thread_rng();
        let dir = Direction::new(rng.gen_range(PI * 0.25..PI * 0.75));

        Self {
            bounds,
            dir,
            velocity: BALL_VELOCITY_INITIAL,
        }
    }

    pub fn bounds(&self) -> &Circle {
        &self.bounds
    }

    pub fn bounds_mut(&mut self) -> &mut Circle {
        &mut self.bounds
    }

    pub fn velocity(&self) -> f64 {
        self.velocity
    }

    pub fn set_pos(&mut self, new_pos: VirtualPosition) -> &mut Self {
        self.bounds.set_pos(new_pos);
        self
    }

    pub fn dir(&self) -> &Direction {
        &self.dir
    }

    pub fn set_dir(&mut self, new_dir: Direction) -> &mut Self {
        self.dir = new_dir;
        self
    }

    pub fn step(&self) -> Circle {
        let shift_x = self.dir.angle().cos() * self.velocity;
        let shift_y = self.dir.angle().sin() * self.velocity;

        Circle::new_unchecked(
            VirtualPosition::new_unchecked(
                self.bounds.pos().x() + shift_x,
                self.bounds().pos().y() - shift_y,
            ),
            self.bounds().radius(),
        )
    }

    pub fn accel(&mut self) {
        self.velocity = (self.velocity * BALL_ACCELL_RATIO).min(BALL_VELOCITY_MAX);
    }
}

impl Default for Ball {
    fn default() -> Self {
        Self::new()
    }
}
