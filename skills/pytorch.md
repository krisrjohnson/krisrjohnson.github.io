---
layout: default
title: PyTorch Resources
---

Official [cheatsheet](https://pytorch.org/tutorials/beginner/ptcheat.html)
jcohnson (port of one of the Style Transfers guy) [toot](https://github.com/jcjohnson/pytorch-examples)

# PyTorch
Install with miniconda, [pytorch.org]("https://pytorch.org") for details.
`conda install pytorch torchvision cudatoolkit=9.0 -c pytorch`

Naming convention, anytime you see `n_something` means the number of that something, such as `n_emb` is the number of embeddings. In Pytorch, and by extension fastai, when you see `somevariable_.someaction()` it's doing in place replacement,

## PyTorch Basics

Some fast.ai lecture 8 takeaways
<!-- the following doesn't actually work -->
{% highlight py linenos %}

	import torch
	a = torch.tensor([1, 2, 3])
	a > 1 #returns [0, 1, 1]
	a = a[a>1]
	a.expand_as((2,3)) # broadcasts to [[1,2,3], [1,2,3]]
	print(a.storage(), a.shape, a.stride())
	a.unsqueeze(0)    # [[1,2,3]] changes a to a rank2 tensor
	a = a[None]           #[[1,2,3]] changes a to a rank2 tensor
	a = a.squeeze()   #gets rid of all unit axes

	b = torch.random(1,3)
	c = torch.zeros(1,3)

	(a < b).float().mean()

	def frob(a): return (a*a).sum().sqrt() #a.pow(2).sum().sqrt()
	def matmul(a,b):
	  ar, ac = a.shape
	  br, bc = b.shape
	  assert ac == br
	  c = torch.zeros(ar,bc)
	  for i in range(ar):
	    c[i] = (a[i, None] * b).sum(dim=0)
	    # c[i] = (a[i].unsqueeze(-1) * b).sum(dim=0)

{% endhighlight %}

Instantiating torch tensors [docs](https://pytorch.org/docs/stable/tensors.html)

`torch.ones(rows,cols), torch.tensor([[1,2][3,4]]), torch.as_tensor(some_numpy_array), torch.randn((d0,..dN)), torch.zeros((d0,..,dN)), a=torch.empty((d0,..,dN)); a.fill_(some_value),`

```python
torch.manual_seed(0) # for reproduceability

a = torch.ones(100,2)
a.size() #100,2
b = torch.tensor([[1,2][3,4]])
x[:,0].uniform_(-1.,1)

z=torch.randn((1, 200,200, 3))
torch.mm(a,b) #np.dot(a,b)
torch.sum(a, dim=-1) #sum tensor over it's last dim, here output shape = 100x1
a.view(d0, d1, d2, d3) #np.reshape([d0,d1,d2,d3]), aka reshapes to whatever dims you specify
a.permute(2,1,0) #BGR -> RGB
torch.mul(a,a) #elementwise multiplication, so this is the Frobenius
torch.matmul(a,b) #if a and b are broadcastable, will perform multiplication
a @ b #ditto above, works in tf now too
```

## Torch img fns
```python
#rgb to bgr
img.permute(2,1,0)

# given list y_hat w/ rgb imgs, y_hat[0].shape = (3,256,256)
y_bgr = [img.permute(1,2,0)[:,:,[2,1,0]] for img in y_hat]
```

[data_preprocessing]: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
[learn_by_ex]: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
[cs231n]: http://cs231n.github.io/

## 1 step model pseudo code
```python
from torchvision.models import vgg19_bn #batchnorm
import torch 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_fn = F.l1_loss #instantiate a class nn.MSE_Loss() or functional nn.F.MSE_Loss
model = vgg19_bn(pretrained=True).to(device) #.eval() for inference, some layers diff behavior
model.zero_grad()
optimizer = torch.optim.adam([params_to_grad], lr=some_lr)
output = model(input)
err = loss_fn(input, target)
err.backward() #calculate gradients
optimizer.step() #apply gradients 
```
## Torch.nn
`Torch.nn.Module()` is the model building block. Can be customized.

### Cuda
When using GPUs, have to move tensors and models over. Doing just `inp.to(device)` will move a copy onto the GPU, so you need to assign it to a new variable and then use that variable. fast.ai had some command to remove/reset everything on the GPU.

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_inp = inp.to(device)

if torch.cuda.is_available(): # alternative, model.to(device) works and is more readable
	model.cuda()
```

### Datasets and DataLoaders
from the docs, data loading and transforms [link](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

Toy example:
```python
class RandomDataset(Dataset): #have to overwrite length and getitem

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)
```

### Hooks
In pytorch you can assign hooks to either the forward or the backward pass

Toy example, takes activations from the ReLUs just before the MaxPool in pre-trained Vgg-19. Used for Style Transfer. Create the hook and register it to the forward pass. When the model's run forward, `activation` will now contain all of the activations for each of those layers.

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vgg_m = models.vgg19(pretrained=True).features.cuda().eval()
blocks = [i-1 for i,o in enumerate(vgg_m.children()) if isinstance(o,nn.MaxPool2d)] #grab
activation = {}

def get_activation(name):
	def hook(model, inp, out):
		activation[name] = out.detach()
	return hook

[vgg_m[i].register_forward_hook(get_activation(f'test-name-{i}')) for i in blocks]
toy_out = vgg_m(toy_inp)

toy_out_activation_layers = []
for layer in activation:
	toy_out_activation_layers.append(activation[layer]) #BOOM
```


### Custom Loss
dummy implementation of nn.MSELoss() is a great example:

```python
def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss

model = nn.Linear(2, 2)
x = torch.randn(1, 2)
target = torch.randn(1, 2)
output = model(x)
loss = my_loss(output, target)
loss.backward()
print(model.weight.grad)
```


## Practical Snippets
Freezing layers in a pretrained model
```python
vgg_19 = models.vgg19(pretrained=True).features.cuda().eval()
#or
for param in vgg_19.parameters():
	param.requires_grad = False

#with children layers inside of layers:
child_counter = 0
for child in vgg_19.children():
	if child_counter < 6:
		for param in child.parameters():
			param.requires_grad = False
	elif child_counter == 6:
		children_of_child_counter = 0
		for children_of_child in child.children():
			if children_of_child_counter < 1:
				for param in children_of_child.parameters():
					param.requires_grad = False
				print(f'child {children_of_child_counter} of child {child_counter} was frozen')
			else:
	child_counter += 1
```


```
