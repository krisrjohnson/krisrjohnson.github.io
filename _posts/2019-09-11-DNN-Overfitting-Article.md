---
layout: single
title: DNN Overfitting
---
_Note: In an effort to curb shallow reading, I'm creating summary posts for articles. My ideal next step is curating a weekly summary post made up of subposts/snippets/what have you._

Summary of Lilian Weng's excellent [post](https://lilianweng.github.io/lil-log/2019/03/14/are-deep-neural-networks-dramatically-overfitted.html)

Summary:
DNNs don’t overfit because they contain a subnetwork (intrinsic dimension) which captures performance while additional parameters go to zero. Models can still be too large but can be pruned back down.

MDL - min desc length - smaller models and mode encapsulations of data are best.

Kolomogov Complexity (K) - smallest program is best, least complex

Network overfitting is not a function of parameters. See figure 4

A 2-D FC model can accurately model any continuous function, with h=2n+d, n is sample, d is dimensions of input. Just the mode will grow extremely wide. Universal approximation theorem. Can be proven by non-singular triangular matrix of middle layer when W is nxd, so WX-B is nxn.

Intrinsic dimensionality - within parameter space D, there is a d that perfectly captures network functionality. Winning lotto ticket and network pruning - train a model then prune weights approaching 0, take the mask of what’s left and retrain only those from scratch and you’ll have an equally accurate model.

Intrinsic dimensionality v2 - if you reinitialize the weights of a given layer of a fully trained model, only the critical layers will result in training error skyrocketing. Re-randomizing weights will break every layer - model turns immediately into random guessing.

