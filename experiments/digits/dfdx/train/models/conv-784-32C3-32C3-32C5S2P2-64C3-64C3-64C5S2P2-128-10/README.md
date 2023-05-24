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

structure inspiration: https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist

training:

```
120.000 images (60.000 source mnist dataset + 60.000 deformed online random generated images)
image deformations: moving least squares, rotate, blur, contrast, scale, shift
optimizer: adam (default)
params datatype: single precision float (f32)
batch size: 32 images
model params: 325.866
~50 epochs
```

testing:

```
10.000 images (source mnist testset)
errors: 85, errors_percent: 0.85%
```