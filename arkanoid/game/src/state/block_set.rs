use crate::primitives::{dimensions::Dimensions, pos::VirtualPosition, rect::Rect};

use super::block::Block;

const BLOCK_ROWS: u32 = 4;
const BLOCKS_IN_ROW: u32 = 10;
const BLOCK_HEIGHT: f64 = 0.05;

pub struct BlockSet {
    blocks: Vec<Block>,
}

impl BlockSet {
    pub fn new() -> Self {
        let mut blocks = Vec::with_capacity((BLOCK_ROWS * BLOCKS_IN_ROW) as usize);
        let block_width = 1.0 / BLOCKS_IN_ROW as f64;

        for r in 0..BLOCK_ROWS {
            let y = r as f64 * BLOCK_HEIGHT;
            for b in 0..BLOCKS_IN_ROW {
                let x = b as f64 * block_width;
                blocks.push(Block::new(Rect::new(
                    VirtualPosition::new(x, y),
                    Dimensions::new(block_width, BLOCK_HEIGHT),
                )))
            }
        }

        Self { blocks }
    }

    pub fn active_blocks(&self) -> impl Iterator<Item = &Block> {
        self.blocks.iter().filter(|b| b.is_active())
    }

    pub fn active_blocks_mut(&mut self) -> impl Iterator<Item = &mut Block> {
        self.blocks.iter_mut().filter(|b| b.is_active())
    }

    pub fn has_active_blocks(&self) -> bool {
        self.blocks.iter().any(|b| b.is_active())
    }
}

impl Default for BlockSet {
    fn default() -> Self {
        Self::new()
    }
}
