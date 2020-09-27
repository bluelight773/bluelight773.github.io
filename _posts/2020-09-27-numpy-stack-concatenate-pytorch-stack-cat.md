# Understanding stack and concatenate

1. TOC
{:toc}

## Concatenation of batches vertically using concatenate and axis=0

Concatenate can be used to join 2 or more arrays along an existing dimension.

Let's say we have two arrays of shape batch size (2) x number of features (3).

~~~python
import numpy as np
a_bf1 = np.array([[1, 2, 3],
                  [4, 5, 6]])
a_bf2 = np.array([[7, 8, 9],
                  [10, 11, 12]])
~~~

Now, let's say we want to join these 2 arrays as to form one array containing 4 rows. In this case, we already have a batch-size dimension and simply want to **expand this existing 0th dimension** from size 2 to size 4. For this scenario, we use `concatenate`. We specify `axis=0` since the batch dimension we want to concatenate along is the 0th dimension. The result is an array of shape 4 x 2. Note that we can apply the concatenation in any of the suggested approaches below.

~~~python
a_cat1 = np.concatenate((a_bf1, a_bf2), axis=0)
a_cat1 = np.concatenate((a_bf1, a_bf2), 0)
a_cat1 = np.concatenate((a_bf1, a_bf2))
a_cat1
~~~
The output is:
~~~python
array[[ 1,  2,  3],
      [ 4,  5,  6],
      [ 7,  8,  9]
      [10, 11, 12]])
~~~

The pytorch equivalent of `concatenate` is `cat`. With `cat`, the `axis` parameter can be specified or not in the same way as with numpy's `concatenate`.
~~~python
import torch
from torch import tensor
torch.cat((tensor(a_bf1), tensor(a_bf2)), 0)
~~~~
The output is:
~~~python
tensor[[ 1,  2,  3],
       [ 4,  5,  6],
       [ 7,  8,  9]
       [10, 11, 12]])
~~~

Another numpy equivalent to `concatenate` with `axis=0` is to use `vstack`. If we think of the 0th dimension as rows, then it makes sense to think of as `concatenate` with `axis=0` as vertically stacking.

~~~python
np.vstack((a_bf1, a_bf2))
~~~
The output is the same as earlier.

~~~python
array[[ 1,  2,  3],
      [ 4,  5,  6],
      [ 7,  8,  9]
      [10, 11, 12]])
~~~

There is no `vstack` in pytorch.

Note that this vertical stacking requires the arrays have the same number of columns, i.e., the 1st dimension should be of the same size so that the arrays can be placed on top of one another vertically.

## Concatenation of batches horizontally using concatenate and axis=1

Let's say we have two arrays of number of batches x number of features with the first array containing the first 2 features, and the second containing the subsequent 3 features for the same rows.
~~~python
a_bf1 = np.array([[1, 2],
                  [3, 4]])
a_bf2 = np.array([[5, 6, 7],
                  [8, 9, 10]])
~~~

Let's say we want to concatenate these arrays as to have one array of shape number of batches (2) x number of features (5). In this scenario, we can use `concatenate` again, but with `axis=1`, since we want to concatenate along the 1st dimension (number of features), ie, we want to **expand the existing 1st dimension**.
~~~python
np.concatenate((a_bf1, a_bf2), 1)
~~~
The output is:
~~~python
array([[1,  2,  5,  6,  7],
       [3,  4,  8,  9, 10]])
~~~

The same could be achieved using pytorch's `cat` or using numpy's helper method, `hstack`. Pytorch has no `hstack` equivalent.
~~~python
print(np.hstack((a_bf1, a_bf2)))
print(torch.cat((tensor(a_bf1), tensor(a_bf2)), dim=1))
~~~
The output is:
~~~python
[[ 1  2  5  6  7]
 [ 3  4  8  9 10]]
tensor([[ 1,  2,  5,  6,  7],
        [ 3,  4,  8,  9, 10]])
~~~

Note that this horizontal stacking requires the arrays have the same number of rows, i.e., the 0th dimension should be of the same size so that the arrays can be placed next to one another horizontally.


## Stacking of instances vertically using stack and axis=0

Now, let's say we have 2 arrays, each corresponding to one instance, and we wish to form one array containing the two rows. In this case, an instance could be the representation of an RGB 2x3 image and thus each array is of shape 3x2x3.

~~~python
a_chw1 = np.array([[[ 1,  2,  3],
                    [ 4,  5,  6]],
                   [[ 7,  8,  9],
                    [10, 11, 12]],
                   [[13, 14, 15],
                    [16, 17, 18]]
                  ])
a_chw2 = np.array([[[19, 20, 21],
                    [22, 23, 24]],
                   [[25, 26, 27],
                    [28, 29, 30]],
                   [[31, 32, 33],
                    [34, 35, 36]]
                  ])
~~~
To form the array containing the 2 images, we need to **insert a new 0th dimension** at the front corresponding to the number of images. To achieve this, we can use `stack` with `axis=0`, since we're introducing a new 0th dimension along which to stack the images. The `stack` function could be called in any of the listed approaches below.
~~~python
a_chw3 = np.stack((a_chw1, a_chw2), axis=0)
a_chw3 = np.stack((a_chw1, a_chw2), 0)
a_chw3 = np.stack((a_chw1, a_chw2))
a_chw3
~~~
The output is the expected array of shape number of images (2) x channels (3) x rows (2) x columns (3).
~~~python
array([[[[ 1,  2,  3],
         [ 4,  5,  6]],

        [[ 7,  8,  9],
         [10, 11, 12]],

        [[13, 14, 15],
         [16, 17, 18]]],


       [[[19, 20, 21],
         [22, 23, 24]],

        [[25, 26, 27],
         [28, 29, 30]],

        [[31, 32, 33],
         [34, 35, 36]]]])
~~~

Alternatively, we can use pytorch's equivalent

## Stacking of instances horizontally using stack and axis=1

<!--
pytorch stack with dim=0
Stacking instances horizontally with stack and axis=1
-->
