---
layout: default
title: Fastai Resources
---


### __Weight decay__:
Use lots of parameters and then penalize complexity. To penalize complexity sum up the value of your parameters. Now that doesn't quite work because some parameters are positive and some are negative. So sum the square of the parameters.

Create a model, and in the loss function add the sum of the square of the parameters. Maybe that number is way too big, so big that the best loss is to set all of the parameters to zero. No good. So to make sure that doesn't happen add the sum of the squares of the parameters to the model and multiply that by some chosen number. Call that number `wd` or weight decay. Take our loss function and add to it the sum of the squares of parameters multiplied by some number wd.

What should that number be? Generally, it should be 0.1. Fastai defaults to 0.01. Why? Because in rare occasions with too much weight decay, no matter how much you train it just never quite fits well enough. However with too little weight decay, you can still train well. Overfitting will happen, so you just have to stop a little bit early.

__forward propagation__: aka the end result after all the fwd steps of a model. E.g., a dense block and a sigmoid fn for a basic classifier, the fwd prop is sigmoid(w^T\*x + b) where w and x are vectors and b is a single bias term.


