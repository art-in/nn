use dfdx::prelude::*;
use dfdx::tensor::AutoDevice;

// https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist
pub type Model = (
    (
        (Conv2D<1, 32, 3>, ReLU, BatchNorm2D<32>), // 32 * (26 * 26) = 15.548
        (Conv2D<32, 32, 3>, ReLU, BatchNorm2D<32>), // 32 * (24 * 24) = 18.432
        (Conv2D<32, 32, 5, 2, 2>, ReLU, BatchNorm2D<32>), // 32 * (12 * 12) = 4.608
        Dropout,
    ),
    (
        (Conv2D<32, 64, 3>, ReLU, BatchNorm2D<64>), // 64 * (10 * 10) = 6.400
        (Conv2D<64, 64, 3>, ReLU, BatchNorm2D<64>), // 64 * (8 * 8) = 4.096
        (Conv2D<64, 64, 5, 2, 2>, ReLU, BatchNorm2D<64>), // 64 * (4 * 4) = 1024
        Dropout,
    ),
    (
        Flatten2D,
        (Linear<1024, 128>, ReLU),
        Dropout,
        (Linear<128, 10>, Softmax),
    ),
);

pub type ModelBuild = <Model as BuildOnDevice<AutoDevice, f32>>::Built;
