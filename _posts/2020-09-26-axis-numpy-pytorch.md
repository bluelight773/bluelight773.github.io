# Understanding the axis parameter in numpy and pytorch

For newcomers to numpy and other data-science-related libraries, such as pytorch, the `axis` parameter can be a source of confusion. It may often appear to do something other than what one may intuitively expect. This blog is an attempt to demystify what it means and how to use it.

1. TOC
{:toc}

## sum, mean, max, min, argmax, argmin with axis=0

The `axis` parameter works in a similar fashion when it comes to many common methods, such as `sum`, `mean`, `max`, `min`, `argmax`, and `argmin`. Our focus here
will be on `sum`, but what is stated regarding sum applies to the other methods mentioned here.

Let's start by forming 2x3 numpy `array`.

~~~python
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])
~~~

Now, let's attempt to apply the sum method with `axis` set to 0. We'll note that this could be done in any of the following ways, but don't run any code just yet.
~~~python
a.sum(0)
a.sum(axis=0)
np.sum(a, 0)
np.sum(a, axis=0)
~~~

Let's try to take a guess at what the code should return. An intuitive guess may be that `axis` 0 corresponds to the rows, and thus we'll be summing the rows and should get something like `array([6, 15)`. These are the sums of the first row and the second row, respectively. Now, let's run the code above and see what we
get.

~~~python
array([5, 7, 9])
~~~

We'll notice that we instead got the sums of the columns. How can make sense of this? One way to do is to think of the `axis` parameter as meaning to collapse the 
0th `axis`. If we collapse the rows on top of one another, we'll have one row where the values from each column will be on top of one another. The 4 will be on top of the 1, the 5 on top of the 2, and the 3 on top of the 6. Then we'll be summing all the values that are on top of one another as to get one row containing
`[5, 7, 9]`. Thus, the key is to think of the `axis` parameter as to mean the collapsing of the axis. It is also worth noting that collapsing the rows is an
indication that the result will have only 1 dimension instead of 2 and that the remaining dimension will be of the same size as the columns in `a`. We can check:
~~~python
s0 = a.sum(0)
print(s0.shape)
~~~
The result is: `(3,)`.

Thus, if we focus on the shapes of the arrays in question. If, as in the example provided, we start with an array of shape `(2, 3)`, collapsing the 0th `axis` to obtain the `sum` or another one of the aforementioned methods, we expect that the result will be of shape `(3,)`. That is, we'll have 3 entries, where each one is the result of the `sum` (or other applied method) of one column.

Thus, in short, **summing with `axis=0` on a 2D array means summing the columns.**

For completeness, we can apply the other computation methods and do using a variety of different ways.
~~~python
print(a.mean(0))
print(a.max(0))
print(a.min(axis=0))
print(a.argmax(axis=0))
print(a.argmin(axis=0))
~~~

The outputs will be the result of computing the `mean`, `max`, `min`, `argmax`, and `argmin`, respectively, across the columns. In all cases, the shape of the
output is `(3,)`.
~~~python
array([2.5, 3.5, 4.5])
array([4, 5, 6])
array([1, 2, 3])
array([1, 1, 1])
array([0, 0, 0])
~~~

Now, let's try an edge case. Let's say we start with a one-dimensional array of shape `(3,)` and attempt to compute the `sum` with `axis=0`. Try to predict the result before running the code.
~~~python
a = np.array([1, 2, 3])
a.sum(0)
~~~
Similar to before, we collapse the 0th dimension, which in this case would mean ending up with the 1, 2, and 3 on top of one another. We then sum the values that
are on top of one another and get just one value, 6. Collapsing the array of shape `(3,)` would suggest removing the 0th dimension, which would subsequently mean
ending up with the shape `()` since there are no dimensions left. We simply have a scalar value left, 6. That is, in fact, the output that we get.

~~~python
6
~~~

While we're at it, it's worth demonstrating that we can apply the same logic and get analogous results when using another library, `pytorch`. Rather than creating numpy `arrays`, we'll use pytorch `tensors`.

```python
from pytorch import tensor
a = tensor([[1, 2, 3],
            [4, 5, 6]])
s = a.sum(0)
print(s)
print(s.shape)
a = tensor([1, 2, 3])
s = a.sum(0)
print(s)
print(s.shape)
```
The output is:
```python
tensor([5, 7, 9])
torch.Size([3])
tensor(6)
torch.Size([])
```
We notice we get results analogous to what we had with numpy. The only difference is in the classes used to represent the inputs and outputs.


## sum, mean, max, min, argmax, argmin with axis=1

