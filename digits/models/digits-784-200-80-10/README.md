network structure:

```
input: 784 (28 * 28 pixel image)

hidden layer 1: 200 neurons
hidden layer 2: 80 neurons
ouput layer: 10 neurons

activation function: tanh
initialization weights: (-1, 1) normal distribution
optimization process: pure stochastic gradient descent
```

training:

```
epoch 1: batches 6k (10 images per patch), learning rate 0.01
epoch 2: batches 6k (10 images per patch), learning rate 0.01
epoch 3: batches 6k (10 images per patch), learning rate 0.01
epoch 4: batches 6k (10 images per patch), learning rate 0.01
epoch 5: batches 6k (10 images per patch), learning rate (linear descent) 0.01 to 0.001
epoch 6: batches 6k (10 images per patch), learning rate (linear descent) 0.01 to 0.001
```

testing (error statistics on test set after epoch 6):

```
count: 10000, errors: 410, errors_percent: 0.041
```