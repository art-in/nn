network structure:

```
input: 784 (28 * 28 pixel image, each pixel in [0; 1] range)

layers:
- convolution (32 output channels, 3x3 kernel, 1 stride, 0 padding), ReLU activation
- batch normalization
- convolution (32 output channels, 3x3 kernel, 1 stride, 0 padding), ReLU activation
- batch normalization
- convolution (32 output channels, 5x5 kernel, 2 stride, 2 padding), ReLU activation
- batch normalization
- dropout (50%)

- convolution (64 output channels, 3x3 kernel, 1 stride, 0 padding), ReLU activation
- batch normalization
- convolution (64 output channels, 3x3 kernel, 1 stride, 0 padding), ReLU activation
- batch normalization
- convolution (64 output channels, 5x5 kernel, 2 stride, 2 padding), ReLU activation
- batch normalization
- dropout (50%)

- flattening
- linear (128 neurons), ReLU activation
- dropout (50%)
- linear (10 neurons), Softmax activation
```

network structure inspiration: https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist

data:

```
two datasets:
- 120.000 images (augmented mnist: 60.000 source samples + 60.000 deformed samples)
- 10.000 images (augmented canvas_digits: 100 source samples + 9.900 deformed samples)

image deformations: moving least squares, rotate, blur, contrast, scale, shift.
deformed samples are generated online, intensity of each deformation chosen randomly.
```

training:

```
optimizer: adam (default)
params datatype: single precision float (f32)
batch size: 32 images
model params: 325.866
~55 epochs (first ~50 on mnist dataset, then 5 more on canvas_digits dataset)
```

testing:

```
10.000 images (on source mnist testset)
images: 10000, errors: 128, error_percent: 1.28%

10.000 images (on x1000 augmented canvas_digits testset)
images: 10000, errors: 26, error_percent: 0.26%
```