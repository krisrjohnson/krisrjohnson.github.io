### Style Transfer from Scratch, Tensorflow

Pseudo Code:
	`Style Image (S) and Content Image (C) to merge
	Use a Gaussian noise image (G) as stock, blank image wouldn't break symmetry?
	Run C and G through a specified layer of pre-trained image classifer, a_C,a_G outputs
	Calculate content loss, the Frobenium norm of Content and Generated images:
		J_C(C, G) = 1/(4*n_H*n_W*n_C) * np.sum((a_C-a_G)**2)
	Run S and G through specific layer of pre-trained image classifier model, a_S and a_G outputs
	Perform non-normalized covariance matrix (Gram) of the channels/filters in the activation layer, G_S,G_G
	Calc style loss, Frobenium norm (Euclidean distance) of the covariance matrices:
		J_S(S, G) = 1/(4*n_H*n_W*n_C) * np.sum((a_S-a_G)**2)
	Calc overall loss, J(C,S,G) = alpha * J_C + beta * J_S
	Perform backprop on the pixels of the Generated image!
	Rinse and repeat`

The Gram matrix is the crux of this implementation. 
A Gram matrix of a set of vectors (v1,...,vn) is the _matrix of dot products_:
	`Gij = vi.T * vj = np.dot(vi,vj)`
If vi is similar to vj, they will have a larger dot product. The diagonal of the Gram, Gii, shows how active each filter is. You could possibly use this to normalize the Gram matrix? 

So the Gram matrix captures the prevalence of different features (Gii) and how often different features co-occur (Gij), aka the style.

Pseudo from coursera ipynb:
1. Create an Interactive Session
1. Load, resize, and normalize w/ VGG means the content image
	- MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
1. Load, resize, and normalize w/ VGG means the style image
1. Randomly initialize the image to be generated
1. Load the VGG19 model
1. Build the TensorFlow graph:
	- Run the content image through the VGG19 model and compute the content cost
	- Run the style image through the VGG19 model and compute the style cost
	- Compute the total cost
	- Define the optimizer and the learning rate
1. Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.


