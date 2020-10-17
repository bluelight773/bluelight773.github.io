# In-Progress: MNIST Logistic Regression Using PyTorch

## Download Data

~~~python
from torchvision.datasets import MNIST
MNIST(root="data/", download=True)
~~~

## Datasets and DataLoaders

~~~python
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, DataLoader

train_and_val_dataset = MNIST(root="data/", train=True, transform=ToTensor())
train_dataset, val_dataset = random_split(train_and_val_dataset, (50000, 10000))
test_dataset = MNIST(root="data/", train=False, transform=ToTensor())

# batch size
bs = 100
train_dl = DataLoader(train_dataset, bs, shuffle=True)
val_dl = DataLoader(val_dataset, bs)
test_dl = DataLoader(test_dataset, bs)
~~~

## Model

~~~python
from torch.nn import Module, Linear

class Model(Module):
  def __init__(self):
    super().__init__()
    # Number of features is number of pixels in this logistic regression model
    # Number of classes is 10 (10 possible digits)
    self.linear = Linear(28*28, 10)
  
  def forward(self, xb):
    # We need to reshape xb to be number of batches x number of pixels
    xb = xb.reshape(-1, 28*28)
    logits = self.linear(xb)
    return logits
~~~

## Train

~~~python
from torch.nn.functional import cross_entropy
from torch.optim import SGD

model = Model()
lr = 1e-4
opt = SGD(model.parameters(), lr = lr)
epochs = 3

for epoch in range(epochs):
  
  # Training
  for xb, yb in train_dl:
    logits = model(xb)
    loss = cross_entropy(logits)
    opt.step()
    opt.zero_grad()

  # Validation
~~~

## Test
