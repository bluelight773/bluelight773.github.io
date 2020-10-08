# Linear Regression Using Gradient Descent From Scratch In PyTorch
This is a tutorial based on [aakshn's](https://jovian.ml/aakashns/02-linear-regression).

Our goal is to build a linear regression model from scratch using gradient descent from scratch using PyTorch.

1. TOC
{:toc}

## x and y
Let's say we have the following dataset:

|Region |Temp. (F)|Rainfall (mm)|Humidity (%)|Apples (ton)|Oranges (ton)|
|-------|---------|-------------|------------|------------|-------------|
|Kanto  |73       |67           |43          |56          |70           |
|Johto  |91       |88           |64          |81          |101          |
|Hoenn  |87       |134          |58          |119         |133          |
|Sinnoh |102      |43           |37          |22          |37           |
|Unova  |69       |96           |70          |103         |119          |

Let's create `x` and `y` tensors corresponding to this dataset.

~~~python
import torch
from torch import tensor

# One row of features per sample (city)
x = tensor([[73, 67, 43],
            [91, 88, 64],
            [87, 134, 58],
            [102, 43, 37],
            [69, 96, 70]], dtype=torch.float32)

# One row per sample (city)
y = tensor([[56, 70],
            [81, 101],
            [119, 133],
            [22, 37],
            [103, 119]], dtype=torch.float32)
~~~

## model
A linear regression model would be of the following form:
~~~python
apples = temp * w11 + rainfall * w12 + humidity * w13 + bias1
oranges = temp * w21 + rainfall * w22 + humidity * w23 + bias2

~~~~

We have 6 weight variables and 2 bias variables. Our goal is to get a good enough approximation of their values, such that we can make reasonably accurate predictions for `apples` and `oranges`.

The weight variables form a tensor of shape `number of ouputs (2) x number of features (3)`. We initialize randomly, but we will improve upon this random initilization later using gradient descent.
~~~python
# Form a 2x3 tensor with random values based on a normal distribution with mean 0 and standard deviation of 1
weights = torch.randn(2, 3, requires_grad=True)
~~~

The bias variables form a tensor of shape `number of outputs (3)`. We initialize randomly, but we will improve upon this random initilization later using gradient descent.
~~~python
# Form a tensor of shape 2 with random values based on a normal distribution with mean 0 and standard deviation of 1
biases = torch.randn(2, requires_grad=True)
~~~

Now, we formulate the `model` function, which computes the predictions.

~~~python
def model(x):
  # We matrix-multiply x by the transpose of weights and add the biases
  return x @ weights.t() + biases
~~~

`x` will be of shape `batch size (5) x number of features (3)`. We transpose `weights` so that it could be multiplied by `x`. The transpose of `weights` is of
shape `number of features (3) x number of outputs (2)`.  Multiplying `x` by the transpose of `weights` produces a tensor of shape `batch size (5) x number of outputs (2)`. We then add the resulting tensor to `biases`, which is of shape `number of outputs (2)`.  Applying element-wise addition of a tensor of shape `5 x 2` to a tensor of shape `2` requires the use of broadcasting since the two tensors are not of the same shape. Broadcasting effectively stretches `biases` to be of shape `batch size (5) x number of outputs (2)` with `bias1` values filling the first column and `bias2` filling the second column.

To obtain an initial set of predictions for `y`, we can simply compute:
~~~python
preds = model(x)
~~~

## loss
In order to improve our `weights` and `biases` values, we need to be able to compute the `loss`, ie, a measure of how far off the actual predictions we happen to be. For a regression problem, we can use the mean squared error as our `loss`. We define our `mse_loss` function as follows:

~~~python
def mse_loss(preds, targets):
  # Compute the squared difference between targets and predictions
  sq_diff = (preds - targets)**2
  # Compute the mean (of the squared error)
  return sq_diff.mean()
~~~

We can now compute the `loss`.

~~~python
loss = mse_loss(preds, y)
~~~

## Compute the Gradient
Now that we've computed the `loss`, we need to adjust our `weights` and `biases` values based on the gradients of the `loss` relative to the `weights` and `biases`. First, we compute the gradients.

~~~python
loss.backward()
~~~

Now, we can look up the gradients using `weights.grad` and `bias.grad`.

## Update Weights and Biases
To apply gradient descent, we adjust the value of `weights` and `biases` by subtracting the corresponding gradients multiplied by a small learning rate. 1e-1 (0.001) is a frequently used value for learning rate. We subtract rather than add because a positive gradient means we should go in the negative direction to reduce the loss and a negative gradient means we should in the positive direction to reduce the loss. We don't want this value adjustment computation to impact future gradient computations, so we use `with torch.no_grad()`.

~~~python
lr = 1e-3
with torch.no_grad():
  weights -= weights.grad * lr
  biases -= bias.grad * lr
~~~

## Putting it Altogether

We've now completed an epoch of calculations whereby we computed our predictions for all `x` values, determined the `loss` then the gradients and adjusted the `weights` and `biases` values accordingly. This completes one epoch. We can create a loop in which all these steps are completed and repeated 100 times. One step we need at the end of each epoch is to zero the gradients so that the current gradients don't accummmulate and impact the future gradients.

Zeroing the gradients can be done as follows. Checking `weights.grad` and `biases.grad` afterwards should output tensors made of only zeros.
~~~python
weights.grad.zero_()
biases.grad.zero_()
~~~

The epoch loop looks as follows:

~~~python
weights = torch.randn(2, 3, requires_grad=True)
biases = torch.rand(2, requires_grad=True)
lr = 1e-3
for i in range(100):
  preds = model(x)
  loss = mse_loss(preds, y)
  loss.backward()
  with torch.no_grad():
    weights -= weights.grad * lr
    biases -= biases.grad * lr
    weights.grad.zero_()
    biases.grad.zero_()
~~~

Upon running the above then checking the values of `preds`, `weights`, or `biases`, we notice that they all consist of tensors containing only `nan` values. We presume this is because our learning rate is too large, and so we make it smaller by an order of magnitude setting to `1e-4`.  The full code will look as follows with the added `torch.manual_seed(1337)` for reproducibility:

~~~python
torch.manual_seed(1337)
weights = torch.randn(2, 3, requires_grad=True)
biases = torch.rand(2, requires_grad=True)
lr = 1e-4
for i in range(100):
  preds = model(x)
  loss = mse_loss(preds, y)
  loss.backward()
  with torch.no_grad():
    weights -= weights.grad * lr
    biases -= biases.grad * lr
    weights.grad.zero_()
    biases.grad.zero_()
~~~

When we check `loss.sqrt()`, we get about `2.4` suggesting our predictions are only off by about 2.4 tons on average.
