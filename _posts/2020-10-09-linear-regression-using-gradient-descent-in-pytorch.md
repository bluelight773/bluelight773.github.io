# Linear Regression Using Gradient Descent In PyTorch

We'll apply linear regression using gradient descent in PyTorch on the Boston Housing Dataset.

1. TOC
{:toc}

## Download Data
Download `Boston.csv` from [here](https://www.kaggle.com/puxama/bostoncsv) into a `data` folder. You can read about the dataset [here](https://www.kaggle.com/c/boston-housing).

## Read CSV
Let's use `pandas` to read in the CSV.

~~~python
import pandas as pd

df = pd.read_csv("data/Boston.csv", sep=",", header=0, index_col=0)
x = df.iloc[:, :-1]

# The target variable is the last column
# Note that using df.iloc[:, -1:] rather than df.iloc[:, -1] ensures 
# we'll have a dataframe rather than a series, which maps to a 2-dimensional array
y = df.iloc[:, -1:]

~~~

## Create TensorDataset
Convert the read-in dataframes into tensors that are passed in to create a `TensorDataset`.
~~~python
from torch.utils.data import TensorDataset
import torch

x = torch.from_numpy(x.to_numpy()).float()
y = torch.from_numpy(y.to_numpy()).float()

ds_train = TensorDataset(x, y)
~~~

## Create DataLoader
Pass in the `TensorDataset` to create a `DataLoader`, which shuffles the rows and places them into batches that we can iterate on.
~~~python
from torch.utils.data import DataLoader

batch_size = 32
dl_train = DataLoader(ds_train, batch_size, shuffle=True)
~~~

## Create a Model
We'll create a model consisting of one linear layer that has as many inputs as the number of features and 1 output (corresponding to 1 target variable).
This linear layer effectively encapsulates the random initialization of weights (one per feature) and of a bias (one for the one output), and applying the
`x @ weights.t() + bias` formula when `x` is passed in.
~~~python
from torch.nn import Linear

model = Linear(x.shape[1], 1)
~~~

## Prepare a Loss Function
Since it's a linear regression model, we'll use Mean Squared Error (MSE) as the loss function.
~~~python
from torch.nn.functional import mse_loss
~~~

## Prepare a Stochastic Gradient Descent (SGD) Optimizer
We need to prepare a Stochastic Gradient Descent (SGD) optimizer that can encapsulate the updating of the model parameters (weights and bias) without impacting the gradients as well as zeroing the gradients. We start with a learning rate of `1e-5`, but we may adjust this if the loss doesn't converge.
~~~python
from torch.optim import SGD

lr = 1e-5
opt = SGD(model.parameters(), lr)
~~~

## Epoch Loop
We can iterate on 100 epochs where each epoch consists of iterating through all the batches. With each batch, we compute the predictions, the loss, the gradients, and then we update the parameters and zero the gradients. 
~~~python
for epoch in range(1, 101):
  for xb, yb in dl_train:
    # Make predictions
    preds = model(xb)

    # Compute loss
    loss = mse_loss(preds, yb)

    # Compute gradients
    loss.backward()

    # Update parameters
    opt.step()

    # Zero gradients
    opt.zero_grad()
    
  if epoch % 10 == 0:
    print("Epoch {}: Loss: {:.2f}".format(epoch, loss.item()))
~~~

## Putting it Altogether
We notice that loss appears as `nan` or `inf` suggesting the learning rate may be too large. Thus, we try a smaller learning rate, such as `lr = 1e-7`, set a manual seed for reproduciblity, and put all the code together. The result is a model with a loss of about `89` corresponding to being off by about `9` on average. 
~~~python
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch.optim import SGD

torch.manual_seed(1337)

df = pd.read_csv("data/Boston.csv", sep=",", header=0, index_col=0)
x = df.iloc[:, :-1]
y = df.iloc[:, -1:]

x = torch.from_numpy(x.to_numpy()).float()
y = torch.from_numpy(y.to_numpy()).float()

ds_train = TensorDataset(x, y)

batch_size = 32
dl_train = DataLoader(ds_train, batch_size, shuffle=True)

model = Linear(x.shape[1], 1)

lr = 1e-7
opt = SGD(model.parameters(), lr)

for epoch in range(1, 101):
  for xb, yb in dl_train:
    # Make predictions
    preds = model(xb)

    # Compute loss
    loss = mse_loss(preds, yb)

    # Compute gradients
    loss.backward()

    # Update parameters
    opt.step()

    # Zero gradients
    opt.zero_grad()
    
  if epoch % 10 == 0:
    print("Epoch {}: Loss: {:.2f}".format(epoch, loss.item()))
~~~
