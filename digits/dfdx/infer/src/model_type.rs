use dfdx::prelude::*;
use dfdx::tensor::AutoDevice;

pub type Model = (
    ((Conv2D<1, 32, 3>, ReLU), MaxPool2D<2>),
    (
        (Conv2D<32, 64, 3>, ReLU),
        (Conv2D<64, 64, 3>, ReLU),
        MaxPool2D<2>,
    ),
    Flatten2D,
    (Linear<25600, 100>, ReLU),
    (Linear<100, 10>, Softmax),
);

pub type ModelBuild = <Model as BuildOnDevice<AutoDevice, f32>>::Built;
