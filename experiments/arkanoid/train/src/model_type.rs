use dfdx::prelude::*;
use dfdx::tensor::AutoDevice;

use crate::game_iterator::{IMAGE_SIZE, PREDICTION_POSITIONS};

pub type Model = Linear<IMAGE_SIZE, PREDICTION_POSITIONS>;

pub type ModelBuild = <Model as BuildOnDevice<AutoDevice, f32>>::Built;
