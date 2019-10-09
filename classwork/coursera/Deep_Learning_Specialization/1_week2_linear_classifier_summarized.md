# Summary Of Binary Cat Classifier
Creating an algorithm which will predict whether or not a cat is in a given picture. Not sure if the input picture has to be of the same size as the sample pics, assume so for now.

## Inputs
Takes say 1000 pictures with a cat, 1000 without a cat, all of tensor 256x256x3. We'll split those into training and testing, with training having say 80% of both, aka 1600 pictures. Each picture in both sets has a corresponding label of 0 or 1, with 1 equating to cat and 0 meaning non-cat.

## Architecture
Our architecture will be a single layer of neurons, all with a single connection (so not dense). We'll sum over all the weights times each pixel and then add in a bias, call this value z. We'll sigmoid this z value to get a prediction in the range [0,1]. .5 or higher we'll take as a prediction of cat in picture, lower than .5 is predicting no-cat.

For updating, we'll use __binary crossentropy__ (although he doesn't say that, at least not yet) loss fn:
`y*log(y_hat) - (1-y)*log(1-y_hat)`, y_hat is the predicted value. I.e. if y==1, loss fn== `log(y_hat)`, y==0 loss fn evaluates to `log(1-y_hat)`. Binary cross entropy penalizes wrong binary predictions while still being a convex function, meaning there is a global minimum of the derivative implying the function can be optimized to a global minimum. This allows us to use gradient descent to optimize our predictions.


## Preprocessing
We'll flatten each picture into a 256x256x3 column. We'll normalize by dividing every pixel by 255 to normalize into [0,1] space. We'll initialize a weight vector of the same length and a single bias term, and set all these values to 0. Pretty sure it'd be computationally more efficient to start our values at random.randn[0,1], but oh well.


## Implementation
Create the forward prop steps, a fn which takes w,b,X,Y, and computes all the way through to the value of the cost fn (aka the average of the loss fn for each X [each picture]) and even partial derivatives of the cost fn for both parameters w and b. This will allow us to perform backpropagation off the output of the fwd prop step.

X doesn't have to be the full set of training or testing, can be a mini-batch. So fn `propagate()` will perform the sigmoid for the batch of input pictures, then calc the cost fn for that batch, and finally output the dw and db. Binary crossentropy on a sigmoid fn has the benefit of being a fairly clean derivative, (Y_hat - Y). The first step of fwd prop is to calc Y_hat, so we can easily use this to calc the partail derivates.

`dw = 1/m * X.dot(Y_hat - Y) #technically a Transpose of that subtraction since Y.shape = (1,m)`<br>
`db = 1/m * sum(Y_hat - Y)`

From those partials, we can perform backpropagation:<br>
`w -= learning_rate * dw`<br>
`b -= learning_rate * db`

Perform for however many epochs, in the implementation this is indeed a for loop.

So our model initializes the weights and bias term. Then for however many epochs loop through fwd computations and backwards propagations to optimize weights and bias toward the lowest cost, all based on train data. Finally do predictions on test data to see test accuracy to get a better gauge of how the model will perform on data it's never seen before.

And that's it!
