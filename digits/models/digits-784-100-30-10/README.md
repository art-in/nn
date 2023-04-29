network structure:

```
input: 784 (28 * 28 pixel image)

hidden layer 1: 100 neurons
hidden layer 2: 30 neurons
ouput layer: 10 neurons

activation function: tanh
initialization weights: (-1, 1) normal distribution
optimization process: pure stochastic gradient descent
```

training:

```
epoch 1: batches 6k (10 images per patch), learning rate (linear descent) 0.01
epoch 2: batches 6k (10 images per patch), learning rate (linear descent) 0.01
epoch 3: batches 6k (10 images per patch), learning rate (linear descent) 0.01 to 0.001
```

testing (error statistics on test set images after epoch 3):

```
count: 10000, errors: 479, errors_percent: 0.0479
```