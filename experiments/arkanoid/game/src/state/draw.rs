use crate::primitives::{circle::Circle, rect::Rect};

use super::State;

pub trait DrawGameState {
    fn clear(&mut self);
    fn rect(&mut self, bounds: &Rect, color: &str, border_color: &str);
    fn circle(&mut self, bounds: &Circle, color: &str, border_color: &str);

    fn draw(&mut self, state: &State) {
        self.clear();
        for block in state.block_set().active_blocks() {
            self.rect(block.bounds(), "#0000ff", "#0000ff") // "#add8e6"
        }
        self.rect(state.platform().bounds(), "#00ff00", "#00ff00"); // "#90ee90"
        self.circle(state.ball().bounds(), "#ff0000", "#ff0000"); // "#f7a88b"
    }
}
