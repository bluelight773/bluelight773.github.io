# Linear Regression Using Gradient Descent In PyTorch

We'll apply linear regression using gradient descent in PyTorch on the Boston Housing Dataset.

## Download Data
Download `Boston.csv` from [here](https://www.kaggle.com/puxama/bostoncsv) into a `data` folder. Read about the dataset [here](https://www.kaggle.com/c/boston-housing).

## Read CSV
~~~python
import pandas as pd

df = pd.read_csv("data/Boston.csv", sep=",", header=0, index_col=0)
x = df.iloc[:, :-1]
y = df.iloc[:, -1:]

~~~


## TensorDataset
~~~python
from torch.utils.data import TensorDataset
import torch

x = torch.from_numpy(x.to_numpy()).float()
y = torch.from_numpy(y.to_numpy()).float()

ds_train = TensorDataset(x, y)
~~~

## DataLoader
~~~python
from torch.utils.data import DataLoader

batch_size = 32
dl_train = DataLoader(ds_train, batch_size, shuffle=True)

~~~

## model
~~~python
from torch.nn import Linear

model = Linear(x.shape[1], 1)

~~~


## loss

~~~python
from torch.nn.functional import mse_loss

~~~

## Optimizer

~~~python
from torch.optim import SGD

lr = 1e-5
opt = SGD(model.parameters(), lr)

~~~


## Epoch Loop

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

~~~python
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch.optim import SGD

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
