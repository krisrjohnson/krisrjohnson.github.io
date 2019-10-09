---
layout: default
title: About KJ
---

# Lecture Notes

## Hyperparameter Tuning Process
In order of importance:
1. Learning rate alpha
1. momentum rate beta
1. number of hidden units
1. mini batch size
1. number of layers
1. learning rate decay
1. Adam: beta1, beta2, epsilon - pretty much never tune

Sample at random for the most important ones. Use a coarse to fine sampling scheme, meaning do a coarse sample then zoom in on a smaller set where the best results were for the first one.

## Appropriate scale for hyperparameters
If you're sampling on the range [0.0001,1], use the log scale to more evenly distribute the scale your searching over.

So sample r in range[a,b], a and b are exponents of 10. And then set 10^r as the parameter you're tuning.

Beta for expnonentially weighted averages can be tricky, as using 0.999 is like averaging over the last 1000 values, .9 is last 10. A linear scale for sampling makes no sense.

Sampling 1-Beta on the log scale is appropriate. So for beta in range[.9,.999], sample r=[-3,-1], (10^-3==.001==1-.999) and use 1-B = 10^r

## Hyperparam Tuning pandas vs caviar
Intuitions do get scale, re-evaluate occasionally

You might train many models in parallel or just babysit one. Pandas vs caviar is a biology reference to reproduction strategies - a single offspring where you invest all your resources or a massive brood where the strong survive. Very application dependent.

## BatchNorm
Fast.ai goes in depth on BatchNorm and especially ResBlocks. From memory, batchNorm renormalizes with a mean and std deviation term the Z, but most importantly also adds a bias term! The bias term actually smooths the gradient and no one could epxlain it for multiple years!

He only mentions the bias term, which per fast.ai is the secret sauce, at the 5 min mark! In all fairness, the normalization effect might speed up training since it's the same as the solution for exploding and vanishing gradients.

You can control the mean and variance of these hidden units by specifying gamma (mean) and beta (variance)[ish?]

## Fitting BatchNorm into a Neural Net
His diagrams aren't great and now he's reusing Beta a crap ton of times.

Basically batchNorm is applied after calculating Z but before performing the non-linearity/activation.

Q: How does this compare to InstanceNorm?

Using mini-batches the mean and std_dev are from that specific minibatch.

Using batchNorm, you can remove the bias term from computing Z since you're going to zero out the mean. BatchNorm adds a bias term. Dimensions of beta and gamma are [n_h,1].

Have to now compute dBeta and dGamma but no longer db. This works gradient descent, momentum, RMSProp, and Adam.


## Why BatchNorm Works!


# Summary

