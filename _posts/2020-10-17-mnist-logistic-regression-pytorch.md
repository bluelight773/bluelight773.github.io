# MNIST Logistic Regression Using PyTorch

We'll apply Logistic Regression on the MNIST dataset. This means we'll have one weight parameter per pixel and one bias parameter per output (class).

1. TOC
{:toc}

## Download Data

~~~python
from torchvision.datasets import MNIST
MNIST(root="data/", download=True)
~~~

## Setup Datasets and DataLoaders

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

## Define Model

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

We tried `lr=1e-4` and obtrained 68% validation accuracy in 10 epochs, then switched to `lr=1e-3` and obtained 84% validation accuracy in 10 epochs.
We also tried `lr=1e-5` and obtained 21% accuracy in 10 epochs, so we stuck with `lr=1e-3`. We achieve 85% test accuracy with `lr=1e-3`.

~~~python
import torch
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch import tensor

def accuracy(logits, labels):
  _, pred_labels = torch.max(logits, 1)
  # It's not necessary, but we wrap value below in a tensor so that it's consistent with
  # cross_entropy (since that returns a tensor)
  return torch.tensor(torch.sum(pred_labels == labels).item() / len(labels))

# For evaluating loss and accuracy on validation and test datasets
def evaluate(model, dl):
    batch_sizes, losses, accuracies = [], [], []
    for xb, yb in dl:
      # No gradient computations when evaluating. Disabling gradients when applying model will
      # ensure gradients won't be tracked in subsequent computations (cross_entropy, accuracy..)
      with torch.no_grad():
        logits = model(xb)
      losses.append(cross_entropy(logits, yb))
      batch_sizes.append(len(xb))
      accuracies.append(accuracy(logits, yb))

    batch_sizes = tensor(batch_sizes, dtype=torch.float)
    losses, accuracies = tensor(losses, dtype=torch.float), tensor(accuracies, dtype=torch.float)
    total = torch.sum(batch_sizes)
    loss = torch.sum(losses * (batch_sizes / total)).item()
    acc = torch.sum(accuracies * (batch_sizes / total)).item()
    
    return loss, acc


def train(model, train_dl, val_dl, test_dl, epochs=10, lr=1e-3):
  torch.manual_seed(42)
  
  opt = SGD(model.parameters(), lr = lr)

  for epoch in range(epochs):
    # Training
    for xb, yb in train_dl:
      logits = model(xb)
      loss = cross_entropy(logits, yb)
      # Compute gradients
      loss.backward()
      # Update weights
      opt.step()
      # Zero gradients
      opt.zero_grad()

    val_loss, val_acc = evaluate(model, val_dl)
    print("Epoch {}: val_loss: {:.4f}, val_acc: {:.4f}".format(epoch+1, val_loss, val_acc))
    
  test_loss, test_acc = evaluate(model, test_dl)
  print("\ntest_loss: {:.4f}, test_acc: {:.4f}".format(test_loss, test_acc))


model = Model()
train(model, train_dl, val_dl, test_dl)

~~~

## Make Predictions

~~~python
def predict(model, img):
  # Make it a batch of 1
  img_b = img.unsqueeze(0)
  # We're not training, so no need to track gradients
  # Disabling gradients when applying model will ensure, they won't be tracked at later
  # steps (eg, softmax, max..).
  with torch.no_grad():
    logits = model(img_b)
  sm = torch.softmax(logits, 1)
  max_probs, pred_labels = torch.max(sm, 1)
  return pred_labels[0].item(), max_probs[0].item()

img, label = test_dataset[0]
pred_label, prob = predict(model, img)
print("Predicted Label: {} ({:.4f}). Correct Label: {}".format(pred_label, prob, label))

~~~

## Save and Load Model

~~~python

torch.save(model.state_dict(), "logistic_regression.pth")
model = Model()
model.load_state_dict(torch.load("logistic_regression.pth"))
evaluate(model, test_dl)

img, label = test_dataset[0]
pred_label, prob = predict(model, img)
print("Predicted Label: {} ({:.4f}). Correct Label: {}".format(pred_label, prob, label))
~~~
