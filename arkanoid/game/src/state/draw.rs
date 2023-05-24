use crate::primitives::{circle::Circle, rect::Rect};

use super::State;

pub trait DrawGameState {
    fn clear(&self);
    fn rect(&self, bounds: &Rect, color: &str, border_color: &str);
    fn circle(&self, bounds: &Circle, color: &str, border_color: &str);

    fn draw(&self, state: &State) {
        self.clear();
        for block in state.block_set().active_blocks() {
            self.rect(block.bounds(), "#add8e6", "#0000ff")
        }
        self.circle(state.ball().bounds(), "#f7a88b", "#ff0000");
        self.rect(state.platform().bounds(), "#90ee90", "#006400");
    }
}
