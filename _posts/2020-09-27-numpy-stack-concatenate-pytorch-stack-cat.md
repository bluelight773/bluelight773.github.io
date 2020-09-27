# Understanding stack and concatenate

1. TOC
{:toc}

## Stack

Let's say we have two arrays of shape batch size (2) x number of features (3).

~~~python
import numpy as np
a_bf1 = np.array([[1, 2, 3],
                  [4, 5, 6]])
a_bf2 = np.array([[7, 8, 9],
                  [10, 11, 12]])
~~~

Now, let's say we want to join these 2 arrays as to form one array containing 4 rows. In this case, we already have a batch-size dimension and simply want to expand it from size 2 to size 4. For this scenario, we use `concatenate`. We specify `axis=0` since the batch dimension we want to concatenate along is the 0th dimension.

~~~python
a_bf3 = np.concatenate((a_bf1, a_bf2), axis=0)
~~~
The output is:
~~~python

~~~


