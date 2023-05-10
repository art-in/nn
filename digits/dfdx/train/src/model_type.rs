use dfdx::prelude::*;
use dfdx::tensor::AutoDevice;

pub type Model = (
    (
        (Conv2D<1, 32, 3>, ReLU), // 32 * (26 * 26) = 21.632
        MaxPool2D<2>,             // 32 * (25 * 25) = 20.000
    ),
    (
        (Conv2D<32, 64, 3>, ReLU), // 64 * (23 * 23) = 33.856
        (Conv2D<64, 64, 3>, ReLU), // 64 * (21 * 21) = 28.224
        MaxPool2D<2>,              // 64 * (20 * 20) = 25.600
    ),
    Flatten2D,
    (Linear<25600, 100>, ReLU),
    (Linear<100, 10>, Softmax),
);

pub type ModelBuild = <Model as BuildOnDevice<AutoDevice, f32>>::Built;
