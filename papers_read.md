---
layout: default
title: Papers Read
categories: [Academic Papers]
tags: [Academic Papers, Deep Learning, Notes]
---

# Scholarly Papers

Make this default behavior list of papers read, w/ summary text collapsed

## Style Transfer
1. Gatys et al. Original: [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)

	Summary: perform style transfer by jointly minimizing feature reconstruction loss vs the content image and style reconstruction loss based on features extracted from a pretrained conv network for the style loss

	_"the key finding of this paper is that the representations of content and style in the CNN are separable. That is, we can manipulate both representations independently to produce new, perceptually meaningful images."_ (pg 4).

	They're effectively pulling out style by taking the different layers of the CNN and doing gram matrix comparisons (gram [explanation](https://towardsdatascience.com/neural-networks-intuitions-2-dot-product-gram-matrix-and-neural-style-transfer-5d39653e7916), youtube [video](https://youtu.be/DEK-W5cxG-g), khan [academy](https://www.khanacademy.org/math/linear-algebra/alternate-bases/orthonormal-basis/v/linear-algebra-the-gram-schmidt-process)). _"Our results suggest that performing a complex-cell like computation at different processing stages along the ventral stream would be a possible way to obtain a content-independent representation of the appearance of a visual input,"_ (pg 9) aka a palette. Use AvgPool instaed of MaxPool, no FC layers or headless (meaning any input img size). A given input img x is encoded in each layer of the CNN by the filter responses to that img.

	Model:
	Have three images, content, style, and noise. For content loss, run content and noise all the way through layer conv5_1 of VGG. Then perform MSE divided by (4 * n_C * n_H * n_W). Style loss, take activations from each conv layer to run through, [1,2,3,4,5]\_1, for noise and style images. Perform MSE on the gram matrices of both for each layer, then divided by (5 * 4 * n_C^2 * n_H^2 * n_W^2). The 5 is to give each layer equal weighting. Total loss equals content loss times alpha plus style loss times beta, with alpha and beta represnting hyperparameters to determine how much weight to give to either choice.

	Backprop: The content loss derivative is straightforward, dLoss equals noise activations minus content activations for all noise activations greater than 0, else 0. The derivative of the style loss per layer is similarly 0 for noise activations less than 0. Otherwise it's the noise image activations transposed times the difference of the noise and style gram matrix outcomes, all divided by the square of the number of channels and pixels (well width and height of the volume at that layer).

	Q's
	- How are they reconstructing the images from the down-sampled layers of the VGG CNN?
	- How would it look if we took layers from different style images? As in conv1_1,2_1,3_1 from style img_1 and conv4_1,5_1 from style img_2? Per their charts, at early conv layers the stylings are very granular. "We find that the local image structures captured by the style representation increase in size and complexity when including style features from higher layers." Why even have the lower layers then, are they the texture? Lower layers look less localized (lol) than higher layers. Higher layers seem to make more sense with regards to the original style.
	- How would we apply this to images without a lot of distortion frame by frame? There's another paper/method that explains a way.
	- Could you stylize a content image with itself? Make Picasso extra Picasso-ey? Testing on Louvre vs. Louvre

1. From Gatys S.T. paper, [Understanding Deep Image Representations by Inverting Them](https://arxiv.org/pdf/1412.0035.pdf)

1. JCJohnson et. al. 2016, [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf), [github](https://github.com/pytorch/examples/tree/master/fast_neural_style)
	Summary: Improvement upon typical image transformation tasks by using a perceptual loss fn (feature loss) instead of per-pixel loss on the output of a feed-forward CNN. Final results are qualitatively worse than Gatys, however feed forward training allows for speedier styling of new images (at 1024x1024 Gatys takes 214s on a GTX Titan X GPU)

	Agree with Fast.ai/Jeremy Howard that feature loss beats perceptual loss for intuitive descriptor. Hell the paper mentions feature reconstruction loss half a dozen times.

	Train an encoder->decoder network using a loss fn that takes a fixed, pre-trained network originally used for image classification, in this case VGG. In effect, training a style encoder that once trained, to stylize a new img, just run through the encoder. Training over different content images makes the encoder content independent as desired. The loss fn is Frobenius norm of relu2_2 for content vs input img, plus gram(frobenius(style - input)) for first 4 ReLUs prior to MaxPool layers, and plus total variation regularization. Each of these three types getting a specific lambda weight as well.

	Because the gram matrix has size CxC, style img input size does not have to equal content or sample image sizes. However, after training when run through with 4x resolution images, can see how styling becomes too granular to look good. Might be symptom of overtraining at low resolution. And because the training data for VGG is class based, VGG featurizes images based on object detection, like looking for eyes, which means unpredictable affects on backgrounds or large blank spaces. Typically the latter gets lightly mosaiced and looks noticeably poor.

	_Total Variation Regularization_, aka total variation denoising, [wiki](https://en.wikipedia.org/wiki/Total_variation_denoising),

	Training: Coco dataset (80k from validation set), resized to 256x256, bs=4, epochs=2, optimizer=Adam

	Could improve by switching VGG to Resnet34. VGG has Maxpool instead of conv2d(stride=2) as well. Could also switch generator structure to a U-Net. For videos, clearly need some RNN structure to capture a relationship to the previous frame. Could a per pixel loss work? Could use newest optimizer over Adam.

	How does this tie to network intrinsic dimensions? Do the last layers of intrinsic dimensions (layers where weight re-initializing affects accuracy/outcome metrics) have what's left of content and structure (per this paper early layers have color, texture, and exact shape, which matches my results looking at activations of passing a cat image through Resnet)



1. want the markov random model mixed S.T.

## Face Puppetry VFX


1. Synthesizing Obama: Learning Lip Sync From Audio, [U Dub paper](http://grail.cs.washington.edu/projects/AudioToObama/siggraph17_obama.pdf)

	With enough audio footage of a speaker, this could be used to accurately synthesize mouth movements, independent of head pose, lighting, etc. However the authors primarily use non-deep learning work arounds to create these results, meaning higher fidelity but also a lot of "cheating."

	Obama did many address the nation, with similar lighting and scene construction, so rich input data.

	Model - audio2shape neural net, seq2seq model - has multiple levels, firstly they train an RNN to yank out mouthshape landmarks given audio (genius idea). Then another model to take as input mouthsape landmarks and output photo-realistic mouth plate. Then some steps to time sync that mouth movement with target video (Q: how do they sink mouth position with head pose? They're generating mouth shape from audio, so it's not connected to head pose, like position and yaw). They're using an LSTM to contain state, plus this paper is from 2017, so should be relatively doable in a day.

	"Simple time-shift recurrent network trained on an uncontrolled, unlabeled visual speech dataset is able to produce convincing results without any dependencies on a speech recognition system."

	On temporal flickering: can use optical flow to interpolate in-between frames.

	Just holy shit on the audio transformation:
		1. input 16kHz mono audio
		1. normalize audio w/ root mean square in ffmpeg [Bellard et al. 2012; Robitza 2016]
		1. Discrete Fourier Transform every 25ms sliding window, 10 ms interval for the slide
		1. 40 traingular Mel-scale filters
		1. take the logarithm
		1. Apply discrete cosine transform - output 13-D vector
		1. combine 13-D vector and log mean energy

	And another holy shit on generating mouth shapes for training:
		1. detect and _frontalize_ Obama's face [Suwajanakorn et al. 2014]
		1. 18 point mouth sheep [Xiong and De La Torre 2013], flatten to 36-D
		1. PCA 36-D and take top 20, per frame
		1. Upsample from 30Hz to 100Hz by linear interpolation (ex, start=10,stop=100, intermediate 9 steps would be +=10 ((100-10)/9))
			1. this is done as audio is in 100Hz, so can have a one-to-one X:Y, only done for training

	They use an RNN because midway through a word mouth height and width are dependent on where they were previously. Specifically, they use a simple layer unidirectional LSTM, multilayer LSTMs did not show marked improvements, with c=60-D and d=20 (200ms). Additionally, they introduce a time delay or 'target delay' by having say x_2 predict y_0, meaning 2 cycle delay as x_0 and x_1 are run through the LSTM and captured by cell memory, c. L2 oss and Adam optimizer.

	Once they have mouth shape, from target video grab possible target frames that best match the shape and perform weighted median on those frames to get a texture for the face. They grab a teeth proxy from the targt video, transferring teeth details directly.

	Candidate frame: landmark detection, estimate 3D pose, and then frontalized and 3Ded using [Suwajanakorn 2014] model. Create an surface to the background plane from the 3D face, using L2. Drift-free 3D pose, apply model onto the frame, and back project the head to the frontal view. Texture is then sythesized based on a pre-determined manual map (face shield and some extra to make bkg blending work better). Clothes masked out using a threshold in HSV (instead of RGB). Now _n_ frames with smalles L2 distance between mouth shape and target mouth shape taken.

	Weighted Median Texture Synthesis: weight is per frame, and comes from exp(LS(mouth_shape_frame, target_mouth_shape) over 2\*sigma\*\*2). Too small sigmas introduce flickering, too large sigmas introduce blur, and is also dependent on number of good frame candidates. So dynamically solve sigma per target shape so weight contribution is a 90% of weight contribution of all available frames.

	Teeth Proxy: Mmmmm no bueno - from process above, media texture provides a good but blurry mask for teeth region while a tooth proxy reference frame _manually_ chosen from the target frames gives obviously higher quality. Teeth are chosen for both upper and lower teeth. The teeth region is found using low-saturation high value thresholding in HSV space within the mouth region defined by the mouth landmarks. Further, a filter gets applied to the selected teeth proxy to basically determine when to use the teeth proxy and when to use the mask. Additionally, to create a photorealistic jawline, a temprally realistic jawline can be created from a weighted average of the landmarks of all image candidates.

	Retiming: Aligning audio and visual pauses so minimal motion during silence. Take target frames without blinking or quick expression changes. To do so, calculate speed of facial landmarks (first derivative) per frame from target video and binary flag blinks by thresholding vertical distance of landmarks of the eyes. Additionally, on the original video, threshold audio volume to create a binary flag for silence.

	Then some optical flow is done with an alpha mask depicting a frontal face to do jaw correction. Finally Laplache pyramid to join all these together.

	Yowza: "For network training, the time for [Suwajanakorn et al. 2014] to preprocess 17-hour Obama video (pose estimation, frontalization) took around 2 weeks on 10 cluster nodes of Intel Xeon E5530. The network training for 300 epochs on NVIDIA TitanX and Intel Core i7-5820K took around 2 hours"


	reference papers:
	1. Supasorn Suwajanakorn, Ira Kemelmacher-Shlizerman, and Steven M Seitz. 2014. Total moving face reconstruction. In European Conference on Computer Vision. Springer, 796–812.
	1. Supasorn Suwajanakorn, Steven M Seitz, and Ira Kemelmacher-Shlizerman. 2015. What Makes Tom Hanks Look Like Tom Hanks. In Proceedings of the IEEE International Conference on Computer Vision. 3952–3960

## NLP

1. Mikolov, [Word2Vec](https://arxiv.org/pdf/1310.4546.pdf)
	Builds off Mikolov's previous work with skip-gram, which was a computationally efficient method for constructing quality word vectorizations from unstructured text data. Good at predicting nearby words. Evaluation is done using analogy reasonings, such as `vec(Montreal Canadiens) - vec(Montreal) + vec(Toronto) => vec(Toronto Maple Leafs)`! Or `vec(Madrid)-vec(Spain)+vec(France) = vec(Paris)`, very nifty. They use phrase based instead of word based, where because this is google based they have tons of text data and must do some rules to pull out phrases. Would guess simple heuristic rules like, accounting for minor words (the,for,and,etc), pull out all consecutive words w/ capitalization in say 20 bn lines of text would give phrases that account for Boston Globe. How did they create this validation set of analogies? How many different word relationships can that capture? Another ex: `quick:quickly :: slow:slowly`

	Skip-gram is bidirectional. They also use hierarchical softmax as more computationally efficient than regular softmax. Regular has on the order Len(W) computations, per word prediction, so usually 10^5-10^7. Hierarchical softmax is on the order Log_2(W) as it implements a binary tree of the output layer. Tree structure effects performace, using binary Huffman tree here.

	Terms: __Noise Contrastive Estimation__, NEGative sampling, hinge loss, cosine distance, and even basic logistic regression used as well.

	They heuristically chose a sampling frequency to discard words like 'the','a','and' etc in the training set.

	". In our experiments, the most crucial decisions that affect the performance are the choice of the model architecture, the size of the vectors, the subsampling rate, and the size of the training window."

1. Word2Vec Parameter Learning Explained [paper](https://arxiv.org/pdf/1411.2738.pdf)

## Nuts and Bolts

1. Hao Li et.al, [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/pdf/1712.09913.pdf)
