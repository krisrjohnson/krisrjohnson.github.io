---
layout: single
title: Style Transfer Through the Ages
categories: [Style Transfer]
tags: [Style Transfer, Deep Learning, Academic Papers, PyTorch]
---

## Style Transfer

Style transfer: rerendering a content iamge in the style of an image or painting, like brushstrokes, shape deformations, or color palette. One of the sexiest applications of Deep Learning (auto generating realistic content has to be CV meets VFX side #1, like re-dubbing movies so the actor fluently speaks say French from English).

Use [arXiv](https://arxiv.org/search/?query=style+transfer&searchtype=all&source=header) to stay up to date on new academic achievements in style transfer.

### __Gatys__ et al. 2015, [abstract](https://arxiv.org/abs/1508.06576):

The original seminal Gatys paper combines two key learnings, intermediate layers of pretrained image classification models hold abstract feature representations of input pictures, and gram matrix transformations of those feature abstractions allows _quantitative_ similarity comparisions. Optimizing similarity comparisons of a blank image against a style image and optimizing direct pixel differences of same blank against a content image creates a hybrid of the content in the form of the style.

Earlier layers of the pretrained network have more literal activations (literal being closer to ground truth) while later layers hold more abstract features, so content activations come from earlier layers and style activaitons from later. The gram matrix of a matrix represents the dot product of all the vectors in the matrix, in this case the vectors are the output activations of each channel. The gram matrix diagonal represents the strength of each filter as it's the filter's cross product with itself, and off-diagonals are cross product strengths of different filter pairings. Google has tons of articles on extracting and visualizing features from filters, here's a good [one](https://distill.pub/2017/feature-visualization/).

_Q/Exp_: Can we mess with the gram matrix outputs to separate styling components? Hypothesis: simple style images such as Chinese Caligraphy or charcoal drawings will have sparse or unimodal gram matrices. Complex paintings will have distinctly different gram matrices.

### __Johnson__ et al. 2016, [abstract](https://arxiv.org/abs/1603.08155):

Primary insight: Can use style image layer outputs as a loss fn for a feed fwd network

JCJohnson et al. create a feed fwd network by taking Gatys' fixed style and content outputs of a pretrained image classification CNN as the loss function to optimize, which they call perceptual loss but I prefer feature loss. By training this feed fwd network on 80k sample images (from msft's Coco) with pre-chosen style weights, they can then pass any new images through the network and output those images restyled in real time. They also include total variation regularizer in the loss function, to "encourage spatial smoothness", aka improve image quality.

The feed fwd network is style image specific and qualitatively isn't as good as using Gatys but the speed ugprade is well worth the trade off.

Once they figure out the hyperparams to style an image how they like, they train a feed fwd network to style automatically for real time stylizations of new images.

### __Duomolin__ et al. 2016, [abstract](https://arxiv.org/abs//1610.07629):

Primary insight: Conditional Instance Norm.

### __ReCoNet__ Gao et al. 2018, [abstract](https://arxiv.org/abs/1807.01197), PyTorch [gitrepo](https://github.com/safwankdb/ReCoNet-PyTorch):

See [video](https://youtu.be/vhBRanZmdH0), which shows how adding in a temporal component renders a smoother styling. It still suffers from edge halo-ing (noticeable color haloes around objects). ReCoNet is basically the JCJohnson et al. Perceptual (feature) Loss network with an added temporal loss, an obvious next step to transition styling to video.

![ReCoNet Architecture](/assets/images/ReCoNet_Architecture.jpg)

Good [article](https://neurohive.io/en/state-of-the-art/real-time-video-style-transfer-fast-accurate-and-temporally-consistent/) explanation

### __AdaIN__ Huang et al 2017, [paper](https://arxiv.org/abs/1703.06868), ex PyTorch [gitrepo](https://github.com/naoto0804/pytorch-AdaIN):

Tuning InstanceNorm to have same mean and variance across content and style images allows a feed forward network to handle _any_ style image, removing previous pretrained model limitations.


## Combining AdaIN with ReCoNet
_Aka, a feed fwd capable of handling any style without frame by frame popping effects_
