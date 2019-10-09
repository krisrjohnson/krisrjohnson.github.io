---
layout: default
title: Numpy Resources
---

# Numpy
[cs231n][cs231numpy] is a decent reference. From python very basics (string formatting, lists, dicts, enum) all the way through SciPy and Matplotlib. Missing things like `np.reshape()` though.

[Numpy documentation](https://docs.scipy.org/doc/numpy/reference/)
Numpy [100 exercises](https://gke.mybinder.org/v2/gh/rougier/numpy-100/master?urlpath=%2Fnotebooks%2F100+Numpy+exercises.ipynb)
Numpy 100 exercises [solutions](https://github.com/rougier/numpy-100)

Numpy, pytorch, and fastai image dimensions: (height, width). PIL and matplolib opposite.

`pip3 install --upgrade numpy`
`import numpy as np`

## The Simple Stuff
more [docs](https://docs.scipy.org/doc/numpy/user/quickstart.html).


ndarray.attributes: .ndim, .shape, .size, .dtype, .dtype.name, .itemsize, .data
Last two aren't really necessary

### Initialize Arrays
```python
# an array
a = np.array([1,2,3][4,5,6]) #initialize 2x3 matrix, aka rank2 array
a.shape

a = np.zeros((3,2)) # 3x2 matrix of 0's
a = np.ones((1,2)) # 1x2 of 1's
a = np.arange(2, 11, dtype=float) #1x10 of 2-10 type float
a = np.linspace(1., 4., 6) #array([1.,1.6,2.2,2.8,3.4,4.])
a = np.full((1,3), 7) #1x3 of 7's

a = np.eye(2) # 2x2 Identity

e = np.random.random((2,2))
x = np.random.randn(64, 100)

a = np.indices((3,3)) #provides transpose
# array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]])

a = np.reshape(a, (2,1))
```

Fun fact, numpy contains euler's constant e and pi: `numpy.e, numpy.pi`

### Slicing/Indexing/Reshaping
```python
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:2, 1:3] #[[2,3], [6,7]]

a[:, 1] != a[:, 1:2] #first expressions is rank1 [2,6,10], second [[2],[6],[10]]

a[np.arange(3), np.array(0,1,0)] += 10 # [[11,2,3,4], [5,16,7,8], [19,10,11,12]]
a > 5 # [[False, False, False, False], [False, True, True, True], [True, True, True, True]]
a[a>5] # [6,7,8,9,10,11,12]

a[-1] # a[-1, :]-> array([9,10,11,12])
# if x.ndim=5, x[1,2,...] == x[1,2,:,:,:]

a.T, a.ravel(), a.reshape(2,6) #return an array, don't modify a
a.resize(2,6) #modifies a
a.reshape(3,-1) #auto calculates the other dims when -1 is given

np.vstack((a,b)) #stacks a on top of b
np.hstack((a,b)) #horizontal

```

### Array Math
numpy does Hadamard operations, aka elementwise, by default
```python
a, b = np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])
a + b # [[6,8], [10, 12]]
a * b # [[5, 12], [21, 32]], etc
np.multiply(a,b) # ditto above
(x / y);np.sqrt(a)

a.dot(b) # matrix multiplication: [[14,17], [30,37]]
np.dot(a,b) #ditto
a@b # ditto

np.sum(a) # sum of all elements, 10
a.sum() #ditto
a.min(), a.max() # 1, 4

a.sum(axis=0) #array([4,6])
a.min(axis=1) #array([1,3])
a.cumsum(axis=1) #array([1,3],[3,7])

np.lingalg.norm(a, ord=2, axis=1, keepdims=True) #array([[5**.5],[5]]) -row-wise vector norms

a.T # [[1,3], [2,4]]
```

### Broadcasting
Alternative [explanation](http://wiki.scipy.org/EricsBroadcastingDoc)
```python
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
# could tile v, np.tile(v, (4,1)), but can straight up add
y = x + v #[[2,2,4],[5,5,6], etc]
```

### Image fns
imread, imsave, imresize:
```python
img = imread('some/file/location.jpg')
img.dtype, img.shape #ex, uint8 (400, 248, 3)
img_tinted = img * [1, 0.95, 0.9] #RGB multiplication using broadcasting (r*1, g*.95, b*.9)
img_tinted = imresize(img_tinted, (300,300)) #resize to 300x300 pixels

import matplotlib.pytplot as plt
plt.subplot(1,2,1); plt.imshow(img)
plt.subplot(1,2,2); plt.imshow(np.uint8(img_tinted))
```

```python
import matplotlib.pytplot as plt
e = np.random.random(480, 240, 3)
e = np.uint8(e*256)
plt.subplot(1, 2, 1)
plt.imshow(img)
```







#### Numpy Full Linear Model - 2-Layer
```python
N, D_in, H, D_out = 64, 1000, 100, 10  #Batch_size, Data in, hidden layer, data out

x = np.random.randn(N, D_in)
y = np.random.randn(n, D_out)
# Dense network, each neuron in first layer is connected to each neuron in next layer, so weight matrix is len(first_layer) * len(next_layer)
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6 #just bc
for t in range(500):
  h = x.dot(w1) #dot is matrix mult in numpy
  h_relu = np.maximum(h, 0)
  y_pred= h_relu.dot(w2)

  #loss
  loss = np.square(y_pred - y).sum()
  print(t, loss)

  #backprop
  grad_y_pred = 2.0 * (y_pred - y)
  grad_w2 = h_relu.T.dot(grad_y_pred)
  grad_h_relu = grad_y_pred.dot(w2.T)
  grad_h[h<0] = 0 #zero out errors?
  grad_w1 = x.T.dot(grad_h)

  #update weights
  w1 -= learning_rate * grad_w1
  w2 -= learning_rate * grad_w2
```


[cs231numpy]: http://cs231n.github.io/python-numpy-tutorial/
