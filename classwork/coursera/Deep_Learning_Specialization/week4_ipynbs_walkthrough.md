plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# DNN Step by Step

### initialize parameters
Start w/ 2 layer model (then generalize to L layer): Linear (Dense) -> ReLU -> Linear (Dense) -> Sigmoid
For matrix multiplication of (row1,col1) * (row2, col2), col1 must be equal to row2, resultant matrix is shape (row1, col2)
```python
def init_paras(n_x, n_h, n_y):
  W1 = np.random.randn(n_h,n_x)*.01
  B1 = np.zeros((n_h,1))
  W2 = np.random.randn(n_y,n_h)*.01
  B2 = np.zeros((n_y,1))
  return {'W1': W1, 'B1': B1, 'W2': W2, 'B2': B2}
```

To generalize, we'd take a list as inputs and loop through the contents to initialize the layers
```python
def init_params(layer_dims):
  params = {}
  for l in range(1, len(layer_dims)):
    params[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * .01
    params[f'B{l}'] = np.zeros((layer_dims[l],1))
  return params
```

### forward propagation
linear forward step -> `def linear_forward(A, W, b): cache = (A, W, b); return np.dot(W, A) + b`

Combining linear forward step and activation step (either sigmoid or relu):
```python
def linear_activation_fwd(A_prev, W, b, activation):
  if activation == 'sigmoid':
    Z, _ = linear_forward(A_prev, W, b)
    A, activation_cache = sigmoid(Z) #provided fn

  if activation == 'relu':
    Z, _ = linear_forward(A_prev, W, b)
    A, activation_cache = relu(Z) #provided fn

  return A, cache
```

L-Layer Model! Create the model that chains all the layers together
```python
def L_model_forward(X, parameters):
  A = X #this is handy for the for loop coming up
  L = len(parameters)/2
  caches = []
  for l in range(1, L):
    A_prev = A
    A, cache = linear_forward_activation(A_prev, parameters[f'W{l}'], parameters[f'b{l}'], activation='relu')
    caches.append(cache)
  Y_hat, cache = linear_forward_activation(A, parameters[f'W{L}'], parameters[f'b{L}'], activation='sigmoid')
  caches.append(cache)
  return Y_hat, caches
```

### Compute Cost (aka loss)
```python
def compute_cost(Y_hat, Y): #Y is (prediction, input_number)
  return -1/(Y.shape[1]) * (np.dot(Y, np.log(AL).T) + np.dot(1-Y, np.log(1-AL).T) )
```

### Backwards Propagation
We know (from calculus) what dZ is, so we'll use it as an input to our backprop function
```python
def linear_backward(dZ, cache):
  A_prev, W, b = cache
  m = A_prev.shape[1]

  dW = 1/m * np.dot(dZ, A_prev.T)
  db = 1/m * np.sum(dZ, axis=0, keepdims = True)
  dA_prev = np.dot(W.T, dZ)

  return dA_prev, dW, db
```

Now link linear_backward with fns that calculate dZ! (dZ[l] = dA[l] * g'(Z[l]))
```python
def linear_activation_backward(dA, cache, activation):
  linear_cache, activation_cache = cache
  if activation == 'relu':
    dZ = relu_backward(dA, activation_cache)
  if activation == 'sigmoid':
    dZ = sigmoid_backward(dA, activation_cache)
  dA_prev, dW, db = lienar_backward(dZ, linear_cache)
  return dA_prev, dW, db
```

Full L-model backwards, takes each layer l's cached fwd values as inputs plus dA! So need to start by calc'ing y_hat's dA!
```python
def L_model_backward(AL, Y, caches); #AL == Y_hat, Y is true outputs, caches is a list of tupled caches per layer
  grads={}
  m = AL.shape[1]
  Y = Y.reshape(AL.shape)
  L = len(caches)
  dAL = - (np.divide(Y,AL) - np.divide(1-Y, 1-AL))
  current_cache = caches[L-1]
  grads[f'dA{L-1}'], grads[f'dW{L}'], grads[f'db{L}'] = linear_activation_backward(dAL, current_cache, activation='sigmoid')
  #Loop from l-2 to l=0
  for l in reverse(range(L-1)):
    current_cache=caches[l]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(current_cache[f'dA{l+1}'], current_cache, activation='relu')
    grads[f'dA{L}'] = dA_prev_temp
    grads[f'dW{L-1}'] = dW_temp
    grads[f'db{L-1}'] = db_temp
  return grads
```

Now to update the parameters, completing our helper fns for our model!
```python
def update_params(parameters, grads, learning_rate):
  L = len(parameters)/2
  for l in range(1, L+1):
    parameters[f'W{l}'] = parameters[f'W{l}'] - learning_rate * grads[f'dW{l}']
    parameters[f'b{l}'] = parameters[f'b{l}'] - learning_rate * grads[f'db{l}']
  return parameters
```

# DNN Application
Taking the fns we've already built, use them to create a DNN! First up is 2-layer
```python
def two_layer_model(X, Y, layer_dims, learning_rate=1e-3, epochs=3000, print_cost=False):
  m = X.shape[1] #expected input is photo flattened into a vector and num training examples, in this case {12288, 209}
  grads = {}
  costs = []
  (n_x, n_h, n_y) = layer_dims
  parameters=initialize_parameters(n_x, n_h, n_y)
  W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
  for i in range(epochs):
    A1, cache1 = linear_activation_forward(X, W1, b1, activation='relu')
    A2, cache2 = linear_activation_forward(A1, W2, b2, activation='sigmoid')

    cost = compute_cost(A2, Y)

    dA2 = - (np.divide(Y,A2) - np.divide(1-Y,1-A2))

    dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation='sigmoid')
    dA0, dW1, db2 = linear_activation_backward(dA1, cache1, activation='relu')

    grads['dW1'], grads['db1'], grads['dW2'], grads['db2'] = dW1, db1, dW2, db2
    parameters = update_params(parameters, grads, learning_rate)
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
    if i % 100 == 0: print(f'epoch: {i}, cost: {cost}')

  return parameters #aka the model, sans architecture!
```
100% accuracy on the training set is surest sign of overfitting available! Test set accuracy of 72% is better than logistic regression at least

Now for the L-Layer model
```python
def L_layer_model(X, Y, layers_dims, lr = 1e-3, epochs=3000, print_cost=False):
  parameters=initialize_parameters(layers_dims)
  for e in range(epochs):
    AL, cache = L_model_forward(X, parameters) # params meta has num layers. Do  fwd prop, each stage added to cache
    cost = compute_cost(AL, Y)
    grads = L_model_backward(AL, Y, caches)
    parameters = update_params(parameters, grads, learning_rate)
  return parameters
```
Remember that grads has dAL hard coded for binary crossentropy, so certainly much more work done to continue to generalize. So now can train a DNN of however many Dense ReLU layers with a final sigmoid layer.


# Summary
We generalized a DNN of form L-1 ReLU layers plus a final sigmoid layer by creating a number of helper functions and then enacting the whole thing in one step. There were some hard coded things in helper functions that would need to be resolved to generalize further. Additionally adding activation type to layers_dims would help further generalize.

The code takes the form:
1. create the weights and biases: `parameters = initialize_params(layers_dims)`
1. loop over epochs
  1. forward pass: `Y_hat, cache = L_model_forward(X,Y, parameters)`
    1. `L_model_forward` has activations hard coded, so first layers are ReLU and last is sigmoid
  1. compute the loss:  `cost = compute_cost(Y_hat, Y)`
  1. do backprop by first calc'ing gradients: `grads = L_model_backward(Y_hat, Y, caches)`
    1. has binary cross entropy hard coded
  1. Update parameters: `parameters = update_params(parameters, grads, learning_rate)`
1. Done! To perform predictions, apply the forward pass using the final set of params

