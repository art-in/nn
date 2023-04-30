network structure:

```
input: 784 (28 * 28 pixel image)

hidden layer: 30 neurons
ouput layer: 10 neurons

activation function: tanh
initialization weights: (-1, 1) normal distribution
optimization process: pure stochastic gradient descent
```

training:

```
epoch 0: batches 1k (10 images per patch), learning rate 0.1
epoch 1: batches 6k (10 images per patch), learning rate (linear descent) 0.01 to 0.001
epoch 2: batches 6k (10 images per patch), learning rate (linear descent) 0.001 to 0.0001
epoch 3: batches 6k (10 images per patch), learning rate (linear descent) 0.01 to 0.001
epoch 4: batches 6k (10 images per patch), learning rate 0.001
```

testing (error statistics on test set after epoch 4):

```
count: 10000, errors: 1001, errors_percent: 0.1001
```