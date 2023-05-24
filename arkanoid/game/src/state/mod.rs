use self::{ball::Ball, block_set::BlockSet, platform::Platform, scene_bounds::SceneBounds};

pub mod draw;

mod ball;
mod block;
mod block_set;
mod platform;
mod scene_bounds;

pub struct State {
    block_set: BlockSet,
    platform: Platform,
    ball: Ball,
    scene_bounds: SceneBounds,
}

impl State {
    pub fn new() -> Self {
        Self {
            block_set: BlockSet::new(),
            platform: Platform::new(),
            ball: Ball::new(),
            scene_bounds: SceneBounds::new(),
        }
    }

    pub fn block_set(&self) -> &BlockSet {
        &self.block_set
    }

    pub fn block_set_mut(&mut self) -> &mut BlockSet {
        &mut self.block_set
    }

    pub fn platform(&self) -> &Platform {
        &self.platform
    }

    pub fn platform_mut(&mut self) -> &mut Platform {
        &mut self.platform
    }

    pub fn ball(&self) -> &Ball {
        &self.ball
    }

    pub fn ball_mut(&mut self) -> &mut Ball {
        &mut self.ball
    }

    pub fn scene_bounds(&self) -> &SceneBounds {
        &self.scene_bounds
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}
