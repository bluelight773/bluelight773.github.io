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

a_2d = np.array([[1, 2, 3],
                 [4, 5, 6]])
~~~

Now, let's attempt to apply the sum method with `axis` set to 0. We'll note that this could be done in any of the following ways, but don't run any code just yet.
~~~python
a_2d.sum(0)
a_2d.sum(axis=0)
np.sum(a_2d, 0)
np.sum(a_2d, axis=0)
~~~

Let's try to take a guess at what the code should return. An intuitive guess may be that `axis` 0 corresponds to the rows, and thus we'll be summing the rows and should get something like `array([6, 15)`. These are the sums of the first row and the second row, respectively. Now, let's run the code above and see what we
get.

~~~python
array([5, 7, 9])
~~~

We'll notice that we instead got the sums of the columns. How can make sense of this? One way to do is to think of the `axis` parameter as meaning to collapse the 
0th `axis`. If we collapse the 0th dimension, which we can think of as the rows in our 2D `array`, we'll have all the rows on top of one another and thus have one row where each value corresponds to the sum of one column. The 4 will be on top of the 1, the 5 on top of the 2, and the 3 on top of the 6. Then we'll be summing all the values that are on top of one another as to get one row containing `[5, 7, 9]`. Thus, the key is to think of the `axis` parameter as to mean the collapsing of the axis. It is also worth noting that collapsing the rows is an indication that the result will have only 1 dimension instead of 2 and that the remaining dimension will be of the same size as the columns in `a`. We can check:
~~~python
s0 = a_2d.sum(0)
print(s0.shape)
~~~
The result is: `(3,)`.

Thus, if we focus on the shapes of the arrays in question. If, as in the example provided, we start with an array of shape `(2, 3)`, collapsing the 0th `axis` to obtain the `sum` or another one of the aforementioned methods, we expect that the result will be of shape `(3,)`. That is, we'll have 3 entries, where each one is the result of the `sum` (or other applied method) of one column.

Thus, in short, **summing with `axis=0` on a 2D array means summing the columns.**

For completeness, we can apply the other computation methods and do using a variety of different ways.
~~~python
print(a_2d.mean(0))
print(a_2d.max(0))
print(a_2d.min(axis=0))
print(a_2d.argmax(axis=0))
print(a_2d.argmin(axis=0))
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
a_1d = np.array([1, 2, 3])
a_1d.sum(0)
~~~
Similar to before, we collapse the 0th dimension, which in this case would mean ending up with the 1, 2, and 3 on top of one another. We then sum the values that
are on top of one another and get just one value, 6. Collapsing the array of shape `(3,)` would suggest removing the 0th dimension, which would subsequently mean
ending up with the shape `()` since there are no dimensions left. We simply have a scalar value left, 6. That is, in fact, the output that we get.

~~~python
6
~~~
Thus, in short, **summing with `axis=0` on a 1D array means summing all the values.**

While we're at it, it's worth demonstrating that we can apply the same logic and get analogous results when using another library, `pytorch`. Rather than creating numpy `arrays`, we'll use pytorch `tensors`.

~~~python
from pytorch import tensor
t_2d = tensor([[1, 2, 3],
               [4, 5, 6]])
s = t_2d.sum(0)
print(s)
print(s.shape)
t_1d = tensor([1, 2, 3])
s = t_1d.sum(0)
print(s)
print(s.shape)
~~~
The output is:
~~~python
tensor([5, 7, 9])
torch.Size([3])
tensor(6.)
torch.Size([])
~~~
We notice we get results analogous to what we had with numpy. The only difference is in the classes used to represent the inputs and outputs.


## sum, mean, max, min, argmax, argmin with axis=1
Now, let's say we set `axis=1`. Try and predict the result of the following before running the code.

~~~python
a_2d.sum(1)
~~~

In this case, we collapse the 1st dimension and thus expect to get a result of shape `(2,)` due to the removal of the 1st dimension (rather than the 0th). Collapsing the 1st dimension, i.e., the columns (rather than the rows), would mean ending up with one columne where the 1, 2 and 3 are on top of one another; and the 4, 5, and 6 are on top of one another. Thus, we'll be summing each row, and getting the values, 6 and 15.

~~~python
array([6, 15])
~~~

Thus, in short, **summing with `axis=0` on a 2D array means summing the rows.**

For completeness, we can apply the other computation methods on a pytorch 2D `tensor`. It is worth noting that in pytorch, the parmeters `axis` and `dim` can be used interchangeably and that we need to use `float` (decimal) values rather than `long` (integer) values in order for the `mean` method to work. Thus, we convert `t_2d` into a `float` tensor.
~~~python
t_2d = tensor([[1, 2, 3],
               [4, 5, 6]]).float() 
print(t_2d.mean(1))
print(t_2d.max(1))
print(t_2d.min(axis=1))
print(t_2d.argmax(axis=1))
print(t_2d.argmin(dim=1))
~~~

The output is shown below. It is worth noting that with pytorch, `max` and `min` return data types that have the attributes `values` and `indices`, each containing the expected tensor for the attribute.
~~~python
tensor([2., 5.])
torch.return_types.max(
values=tensor([3., 6.]),
indices=tensor([2, 2]))
torch.return_types.min(
values=tensor([1., 4.]),
indices=tensor([0, 0]))
tensor([2, 2])
tensor([0, 0])
~~~


<!--
Can't apply axis=1 on 1D array
Add appying axis=-1 in case of axis=1
Work on 3D array
a=np.array([
            [(1,2,3), (4,5,6)],
            [(7,8,9), (10,11,12)]
           ])
Provide multiple axes eg axis=(0,1) makes sense in the rowsxcolsxchannels example
Work on stack
-->
