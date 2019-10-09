---
layout: default
title: About KJ
---

# Lecture Notes

## Mini-batch Gradient Descent
Nothing really new here, use mini-batches to drastically speed up training time. From practice, need to know when is too few for mini-batch.

## Understanding mini-batch gradient descent
The cost function is not guaranteed to decrease on every mini batch iteration. Because you're training on a different set of data each batch, variance in the distribution of each mini-batch means you can have higher cost batches.

Too small of mini-batch means high variance on the gradients, which can cause training to take longer.

Typical minibatch sizes are [64,128,256,512] - this video is old; really it's can your mini batch fit in GPU memory

## Exponentially Weighted Averages
Andrew Ngu is from London! Fun fact.

If you have ordered data, such as time data, you can set the rolling average data to be `V[t] = Beta * V[t-1] + (1-Beta)Data[t]`

Because V[t-1] is dependent on V[t-2] and on and on, you're still slightly representing earlier datapoints! Can rewrite as:
    `V[t] = (1-Beta)Data[t] + Beta.pow(1) * Data[t-1] + Beta.pow(2) * Data[t-2] + Beta.pow(3)Data[t-3]...`
Hence you're averaging, but with Beta exponentially multiplying, aka decaying.

## Understanding Exponentially Weight Averages
Basic rule for how many days you're averaging over is what power of Beta makes Beta.pow() ~~ 1/e? For Beta=.9, that'd be 10. Beta=.98 it'd be 50.

In practice, cache V[theta] and update as you go forward. Aka, new V[theta] = V[theta] + (1-Beta) * Theta

## Bias Correction in Exp Weight Avgs
The initial phase has issues, as V[0]==0, so V[t=1] = V[t=0] + (1-Beta)Theta[1]. Meaning V[t=1] will be close to 0 as Beta is usually around .9.

This can be corrected by dividing V[t] by (1-Beta.pow(t)). Since usually .9 < Beta < 1, this will yield a small number which will counteract the (1-Beta) in the numerator, while t is small. As t grows, the denominator will approach 1 and therefore disappear.

In practice bias correction is not done

## Momentum
__Momentum__ = exponentially weighted average applied to the gradient.

In effect, smooths the gradient, which without exponentially weighted averaging will have variance in non-global minimum directions. Acts as a dampener on the variance of gradient descent.

`VdW = Beta * VdW +(1-Beta) * dW`, aka new VdW equals previous VdW plus one minus Beta times gradient.

## RMSProp
__Root mean square prop__: Compute dW, db on current mini batch. Using momentum, because if you didn't then these computations would actually zero out, you square the current batches dW term. So the `VdW = Beta*VdW + (1-Beta)*dW^2`. For nomenclature reasons, VdW we'll now call SdW:
```
SdW = Beta*SdW + (1-Beta)*dW^2
W = W - alpha * dW / SdW^.5
```

So it's an exponentially weighted average of the square of the derivatives. And on the W update, you divide the update by the square root of the exponentially weighted average.

This further dampens oscillations. So can use a larger learning rate to get faster convergence!

Must add a small epsilon to the denominator of the W update step so you don't divide by an SdW close to 0.

## Adam Optimzation Algo
__Adam Optimizer__: The combo of momentum and RMSProp. Stands for Adaptive Momemt Estimation.

Initialize: VdW = 0, SdW = 0, (etc w/ B terms)

On iteration t:
1. Copmute dW, db using current minibatch
1. `VdW = Beta1*VdW + (1-Beta1)*dW; SdW =Beta2*SdW + (1-Beta2)dW^2`
1. Do bias correction on both, aka divided VdW by (1-Beta1^t), etc
1. `W := W - alpha * VdW/(SdW^2 + epsilon)`

So adding some hyperparameters, learning_rate still important, Beta1 is typically .9, Beta2 typically .999, and epsilon 10e-8

## Learning Rate Decay
Fast.ai uses fit_one_cycle to do this "smarter", but the basic idea is to slow your learning rate as you get closer to convergence.

`LR = LR_0/(1+decay_rate*epoch_num)`

E.g., with a decay rate of 1 and alpha_0 = .2, LR[0:5] = [1/2, 1/3, 1/4, 1/5]

Other decay methods:
`alpha = alpha_0 * .95 ^epoch_num` - exponential decay
`alpha = alpha_0 * k/epoch_num^.5`
Or even just dividing by 2 repeatedly after a set number of examples.

## The problem of local Optima, aka saddle points
For CNNs, resblock "solves" this even further.

For DNN, in say a 20k dimension space, __all__ the points would have to have 0 gradients for a local minimum. This is extremely unlikely.

Ngu drew a dinosaur when he tried to draw a horse!!

Real problem is a plateau, where derivatives are very small or close to zero for a long time. Momentum and RMSProp help speed up training during plateaus.

# Summary
Mini batches are necessary and commonplace at this point, training data is usually just too large to all fit in memory. Exponentially weighted averages, aka momentum, help smooth gradients. Combined with RMSProp is called the Adam optimizer, which is short for Adaptive Moment Estimation.

Tuning the learning rate can also improve training times. Learning rate decay slows down learning as you hopefully approach convergence. Momentum and RMSProp help with getting past plateaus faster. Global minima in such high dimensional spaces are incredibly unlikely so not a problem.
