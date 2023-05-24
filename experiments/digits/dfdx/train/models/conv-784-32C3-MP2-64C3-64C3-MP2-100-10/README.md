network structure:

```
input: 784 (28 * 28 pixel image, each pixel in [0; 1] range)

layers:
- convolution (32 output channels, 3x3 kernel, 1 stride, 0 padding), ReLU activation
- maxpool (2x2 kernel, 1 stride, 0 padding)
- convolution (64 output channels, 3x3 kernel, 1 stride, 0 padding), ReLU activation
- convolution (64 output channels, 3x3 kernel, 1 stride, 0 padding), ReLU activation
- maxpool (2x2 kernel, 1 stride, 0 padding)
- flattening
- linear (100 neurons), ReLU activation
- linear (10 neurons), Softmax activation

optimization process: stochastic gradient descent (0.01 learning rate, 0.9 momentum)
params datatype: single precision float (f32)
```

training: ~30 epochs

testing (error statistics on test set):

```
count: 10000, errors: 84, errors_percent: 0.84%
```