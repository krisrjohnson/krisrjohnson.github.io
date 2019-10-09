# Shallow Neural Network

## Lecture notes

### Computing a Neural Network's Output
The notation he uses is different from fastai's, he's using a neuron to represent both the weight embeddings and the activation fn, fast.ai lists them all out and makes it a point to denote embeddings vs activations/parameters. Embeddings being numbers you're training and activations being numbers calculated from a fn.

So a single hidden layer NN with hidden layer having 4 nodes, Ng describes
1. x1,1..3 as inputs
1. w1,1..4 as cols for each neuron,
  1. .shape==4x3, aka the weights for input x1 are the first column, [w11,w21,w31,w41].
1. b1,1..4 as bias terms for each neuron .shape==4x1
1. a1,1..4 is sigmoid on result of (w1.T * x + b1) .shape==4x1
1. w2,1 another weights layer, of a single col of 4 terms .shape==4x1
1. one bias term for this single weights col, b2 .shape==1x1
1. a2 is sigmoid result of (w2.T * x + b2) .shape==1x1
1. a2 == y_hat

Fastai would describe this as:
1. input layer x1,1..3 .shape==3x1
1. w1,1..4 as embedding layer .shape==4x3x5
  1. each embedding col is 5 terms, the bias plus 4 weights
1. a1,1..4 as a sigmoid activation layer
  1. fast.ai automatically handles the separate calcs performed on bias and weights
1. w2,1 another embedding of single col of 5 terms, first is bias
1. a2 is output from sigmoid activation

### Vectorizing across multiple examples
Basically, for your inputs, .hstack(). Aka, turn each input into a column and stack them next ot each other.

Then you don't have to transpose the weight matrix. In this example, X is now of shape 3xm and W is 4x3 so matrix multiply will be 4xm. The bias terms must be broadcast to 4xm though!

So training examples are stacked horizontally throughout all layers, as in the col corresponds to the training example.

### Explanation for Vectorization Example
Straightforward, I recapped above

### Activation Functions
1. sigmoid = `1/(1+e^-z) = e^z/(1+e^z)`
1. tanh = `(e^z - e^-z)/(e^z + e^-z)`, range: [-1,1];
  1. it's a shifted and scaled version of the sigmoid
  1. Usually has better results than sigmoid, especially as activations centered around 0 which is typically the mean of our input data
  1. If z >> 0 or z << 0, slows down learning since slope becomes very flat
1. ReLU - Rectified Linear Unit: `a = max(0,z)`
  1. technically not continuous, so not differentiable, but in practice doesn't matter
  1. Leaky: `a = max(0.01z, z)` non-zero slope for negative values

ReLU everywhere except [0,1] binary classificaiton output layers where you'd use sigmoid.

### Why you need non-linear activaiton fns
fastai - result of affine on affine is still just an affine. Need non-linear to capture the curvy bits ~ Jeremy Howard

Same explanation here. Linear of linear is linear, so can't be more expressive than just linear. Only time you might use linear activation fn is linear regression, such as linearly predicting housing prices.

### Derivatives of Activaiton Fns
sigmoid derivative = sigmoid(1-sigmoid). Proof-ish:

sigmoid=1/(1+e^-z); sigmoid' = d(1+e^-z)-1 = -1(1+e^-z)^2 dz[(1+e^-z)] = -1(1+e^-z)^2 * e^-z * (-1) ->
(e^-z)/(1+e^-z)^2 = 1/(1+e^-z) * (e^-z/(1+e^-z)) -> sigmoid * ((1+e^-z) -1)/(1+e^-z) -> sigmoid((1+e^-z)/(1+e^-z) -1/(1+e^-z)) -> sigmoid * (1-sigmoid)

tanh derivative = 1-tanh^2

ReLU = 0 if z<0, 1 if z>0, set it to 0 if z=0
Leaky ReLU = 0.01 if z<0, 1 if z>0, 0 if z=0

### Gradient Descent for Neural Networks
One hidden layer

Similar to week2, gradient descent is the partial derivative with respect to the cost function for each parameter. With one hidden layer, there are four parameters, W[1], B[1], W[2], B[2]. So to update W[1] using gradient descent with a cost function J(W[1],B[1],W[2],B[2]), W[1] -= Learning_Rate * dJ/dW[1].

For this binary classification problem:
```
#forward prop:
Z[1] = W[1]X+B[1]
A[1] = g[1](Z[1])
Z[2] = W[2]A[1]+B[2]
A[2] = g[2](Z[2]) = sigmoid(Z[2])

#backprop:
dZ[2] =A[2]-Y #sigmoid der
dW[2] = 1/m *dZ[2]A[1].T
dB[2] = 1/m *np.sum(dZ[2], axis=1, keepdims=True)
dZ[1] = W[2].dZ[2] *g'[1](Z[1])
dW[1] = 1/m * dZ[1]X.T
dB[1] = 1/m * np.sum(dZ[1], axis=1, keepdims=True)
```

### Backprop Intuition
dW[i] =1/m * dz[i]A[i-1].T
dB[i] =1/m * np.sum(dz[i], axis=1, keepdims=True)

### Random Initialization
Initializing bias terms to all 0's is okay but not weights as backprop would be symmetrical. Weights initialized to small values, [0,0.01], in the high slope range so they can move in the proper direction asap.

## Quiz
All that are true:
1. a[2](12) denotes the activation vector of the 2nd layer for the 12th training example.
1. a[2]\_4 is the activation output by the 4th neuron of the 2nd layer
1. a[2] denotes the activation vector of the 2nd layer.
1. X is a matrix in which each column is one training example.

The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer. True/False? A: True

Which of these is a correct vectorized implementation of forward propagation for layer l, where 1 <= l <= L1?
Z[l] = W[l] * A[l-1] + b[l]
A[l] = g[l](Z[l])


For binary classifier, what activation fn would you use for output layer? Sigmoid
```python
A = np.random.randn(4,3)
B = np.sum(A, axis = 1, keepdims = True)
```
What shape is B? A: 4x1. Keepdims=True means it'll have 1 col instead of .shape being (4,)

Suppose you have built a neural network. You decide to initialize the weights and biases to be zero. Which of the following statements is true? A: Each neuron in the first hidden layer will perform the same computation. So even after multiple iterations of gradient descent each neuron in the layer will be computing the same thing as other neurons.

Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False? True


You have built a network using the tanh activation for all the hidden units. You initialize the weights to relative large values, using np.random.randn(..,..)* 1000. What will happen? A: This will cause the inputs of the tanh to also be very large, thus _causing gradients to be close to zero. The optimization algorithm will thus become slow._


## Summary
Single hidden layer is very similar to logistic regression. Just more backpropagation steps. The linear algebra works out nicely when we stack our inputs, x_i, horizontally into an input matrix X. Embeddings aka weight matrices are of shape (neurons in output layer, neurons in input layer).

In practice we largely use ReLU activation functions or even tanh as it's [-1,1] range is better suited to data normalized around 0.
