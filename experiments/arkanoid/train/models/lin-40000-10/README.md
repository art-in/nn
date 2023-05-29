network structure:

```
input: 40000 (100 * 100 pixel image, pixel is RGBA channels, channel in range [0,1])

layers:
- linear (10 neurons)
```

training:

```
10.000 images in each epoch. each epoch initiates new game, and iterates that game until
certain number of steps, i.e. certain number of images are generated, not stopping on fail
or win. loss is difference between predicted position of platform and its optimal position,
where optimal position of platform equals to ball's X position, i.e. teaching model to hold
platform under the ball all the time. inference output is one of 10 possible X positions of
the platform.

all objects are filled with monotonic color. each object type utilizes its own color channel.
ball is red (#ff0000), platform is green (#00ff00), block is blue (#0000ff).

optimizer: adam (default)
params datatype: single precision float (f32)
batch size: 32 images
model params: 400.010
~100 epochs
```

testing:

```
1.000 images
errors count is unstable due to randomized starting direction of the ball.
most of runs errors are under 1%
```