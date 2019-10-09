
##

# Notes

## Setting up your ML application
### Train/Dev/Test Sets
layers,
hidden units, learning rates, activation functions
Cycle: Idea -> Code -> Experiment -> Idea
Training data: Data used to actually train the model
Dev/Validation Set: See how model performs on data it wasn't trained on
Testing Set: for final model accuracy

With explosion of data collection, dataset can be 1m+, and only need say 10k for dev and test sets, meaning 98/1/1% splits across train, dev, and test sets.

Example, training set crawled from the web while dev/test sets are from users uploading pics from phones.

Dev and test sets should always come from the same distribution.

If only have train/test set, you really only have train/dev set and are overfitting to the dev set. Lots of time can be okay.

### Bias and Variance
__"I've noticed that almost all the really good machine learning practicioners tend to be very sophisticated in understanding Bias and Variance. Bias and Variance is one of those concepts that's easy to learn but difficult to master"__

Two key metrics for bias and variance are Train set error and Dev set error

Low training set error with high dev set error is high variance. High training set error and similarly high dev set error is high bias and is underfitting. High training set error with very high dev set error is high bias and high variance. Predicated on optimal error (Bayes error) is ~ 0%.

### Basic Recipe for dealing w/ High Bias or High Variance
High bias - training data performace - high error
  1. bigger network
  1. traing longer
  1. (NN architecture search)
High Variance - dev set performance
"Are you able to generalize from having a pretty good training set performance to having a good dev set performance"
  1. more data
  1. Regularization
  1. (NN architecture search)

Pre-deep learning era, bias-variance tradeoff meant improving one hurt the other. Nowadays, bigger networks and more data will improve without hurting.

## Regularization Tools
### Regularization
Weight decay, aka L2 regularization, means adding the Frobenius norm of the weight matrix (for logistic regression) scaled by a hyperparameter lambda to y_hat. So W - alpha * dW will include a regularization term in dW, increasing how much is removed from W. Aka, the weights will decay.

### Why does Regularization help with overfitting
Using L2 regularization, aka weight decay, will zero out activations which are rarely used, whereas before those activations would become overfitting. Weight decay 'zeroes out' hidden units (makes them tiny).

Another intuition, if lambda is large then W and Z will be small. With a tanh activation function that means A=tanh(Z) will be in the roughly linear range, so the final form will be compounding roughly linear segments instead of higher polynomial shape curves, removing overfitting.

### Dropout Regularization
_dropout_: drop a neuron with a hyperparameter probability, per batch.

_Inverted dropout_: keep_prob might be 0.8, so going to eliminate 20% of neurons. For say layer 3, `d3 = np.random.rand(a3.shape[0], a3.shape[1])` and `a3 = np.multiply(a3,d3)`, meaning you zero out the 20% of a3. Finally you divide a3 by the keep_prob parameter to correct the expected value, this division is where the name inverted comes from.

Do these zeroed out weights get affected by gradient descent? Or do we keep the d3 and hence a3 layer?

__Important__: At test time no drop out!! Also, don't zero out single neuron layers!

Drop-out's effect is similar to L2, it shrinks the weights.

The cost function is now no longer well defined. Checking performance is now harder, removes the J should monostically decrease debugging tool Worth running with keep-dims = 1.0 to test J decreases with every iteration as a debug, then running w/ dropout

### Other regularization methods
For reducing overfitting

_Data Augmentation_, aka _transforms_: flip vert, flip horizontal, warp, morph, random crop/distortion.

Obviously don't add as much information as a whole new datapoint, but still an inexpensive way to give algorithm more data. Basically telling the algorithm that say cats are vertically symmetrical.

_Early stopping_: plot J or classification error as you train, and on dev set, and make sure to stop before dev set error starts to increase (meaning your algo no longer generalizes as well).

Downside: couples optimizing J fn and not overfitting. Instead of using different tools to solve these two problems, ala orthogonalization.

Ngu says it's easier to just use lots of different values of lambda for L2 reg and avoid early stopping.

## Setting up your optimization problem
### Normalizing inputs
Helps speed up training.

2 steps - subtract/zero out the mean for every training example. Then normalize the variance by dividing out the sigma per feature.

Must use the same means and std deviations on the test set!

If you don't normalize, the cost function may be very elongated, making gradient descent more difficult. Whereas with normalization, graident descent will be more uniform for however you initialize your parameters, meaning larger learning_rates and approaching the global minimum faster.

Guarantees all your features will be similar scales which will speed up training the algorithm.

### Vanishing/Exploding gradients
For very very deep NNs, gradients can become very very big or exceedingly small, because weights in later layers increase or decrease exponentially.

### Weight initialization for deep networks
The partial solution for exploding/vanishing weights is better initialization of parameters.

Want variance of W[i] to be 1/n. So `np.random.randn(shape) * np.sqrt(2/n[l-1])` With ReLU, want Var(W[i]) =2/n. Basically this keeps Z close to 1, so for deep layers will stop weights from exponentially growing or disappearing.

For tanh, you want Xavier initialization. `Var = (2/(n[l-1]*n[l]))^.5`

This hyperparameter is lower down in the hierarchy of experimenting with hyperparameter levers.

### Numerical Approximation of gradients
f(theta) = theta^3
It's easier to get a derivative approximation using theta+epsilon and theta-epsilon instead of just theta+epsilon alone. The error of this approximation is on the order of epsilon^2, which for epsilon < 1, epsilon^2 is smaller than just epsilon

By taking a two sided difference, you can numerically verify whether or not a function g, g(theta), that someone else gives you is a correct implementation of the derivative of a function f

### Gradient Checking
Take all your parameters and reshape into a giant vector theta (reshape and concatenate), making J(theta). Reshape all the gradients into giant dTheta vector.

for each training sample i, dTheta_approx[i] = (J(theta_i+epsilon) - J(theta_i - epsilon))/2epsilon

Then check if dTheta_approx ~~ dTheta[i] by calculating the euclidean distance over the sum of the euclidean lengths.

In practice, use epsilon=10e-7. An output of 10e-7 or smaller then fine. If 10e-5 then recheck. 10e-3 much more concerned there's a bug.

### Gradient Checking Implementation Notes
- Only use in debug, backprop for dTheta
- Look at components to try to id bug
  - look at which values dTheta_i are very far off
- Remember regularization when doing dTheta
- Grad check doesn't work with dropout, `keep_prob = 1.0`
- (Rare) Run grad check at rand initialization, then run after some number of iterations

# Summary
Setting up train/dev/test, dealing with bias and variance (bias is train performance, variance is dev performance), regularization - dropout, L2/weight decay, tricks for speeding up training - normalizing inputs, and debugging with gradient checking.

Initalizing all params to zeros will cause fn to be perfectly symmetrical, and all weights will learn at the exact same rate, so the algo will be broken.

Initializing to large params will cause high gradients and learning rates, so training will take longer
