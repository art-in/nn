network structure:

```
input: 784 (28 * 28 pixel image)

hidden layer: 100 neurons
ouput layer: 10 neurons

activation function: tanh
initialization weights: (-1, 1) normal distribution
optimization process: pure stochastic gradient descent
```

training:

```
epoch 0: batches 3k (10 images per patch), learning rate (linear descent) 0.01 to 0.001
epoch 1: batches 6k (10 images per patch), learning rate (linear descent) 0.01 to 0.001
epoch 2: batches 6k (10 images per patch), learning rate (linear descent) 0.01 to 0.001
epoch 3: batches 6k (10 images per patch), learning rate (linear descent) 0.01 to 0.001
epoch 4: batches 6k (10 images per patch), learning rate (linear descent) 0.01 to 0.001
epoch 5: batches 6k (10 images per patch), learning rate (linear descent) 0.01 to 0.001
```

testing (error statistics on test set images after epoch 5):

```
count: 60000, errors: 5407, errors_percent: 0.09011666666666666
```