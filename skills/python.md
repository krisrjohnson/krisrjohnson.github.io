---
layout: default
title: Python Cheatsheets
---

# Python
I'll probably get the most use out of this by further splitting it up. Will do so if the size of this page becomes untenable.

Good tuts/blurbs/walkthroughs:
1. [decorators](https://www.codementor.io/sheena/advanced-use-python-decorators-class-function-du107nxsv)
1. [metaclasses](https://realpython.com/python-metaclasses/)


## OS and I/O
Useful pkgs: os, argparse, pathlib
```python


```
## Lists

### Basics
```python
list_var = [1,2,3, '4','delta', 'united']
list2= ['a','e','i','o','u']
var_1, var_2, _ = *list_var #var_1=1 and var_2=, * unpacks the list
del list_var[2:4] #removes 3,'4'
list_var.insert(2, 3)
list_var.extend(list2) #tacks elems of list2 to end of list_var
for i,x in enumerate(list_var): print(i,x) #adds an iterator counter, i
```
Functions:
- list.append(element) - add element to end of list
- list.extend(list) - add all elements of a list to another list
- list.insert(index, element)
- del list[start_index:end_index] #as always end_index is not included
- list.sort() #modifies list in place! can be memory intensive
- list.reverse() #ditto
- list.pop(index)
- list.clear() #empties list
- list.copy()
- list.count(elem)
- list(filter(fn, list2)) # returns list where fn(list2[i]) is True
- np.round(list_var)


### List comprehension
Basic idea:
```python
vals = [expression
        for value in collection
        if condition]

# This is equivalent to:
vals = []
for value in collection:
    if condition:
        vals.append(expression)
```

type annotations:
```python
def my_add(a: int, b: int) -> int: return a + b
```

## Objects
Almost everything in python is an object.

Fun fact: `self` is used by convention for the instance reference within class init etc, but can actually be name anything.

### Object Introspection
How to know what methods an object has. getattr, hasattr, other fns. [Useful link][dive_into_python_ch4], [IBM 2002 link][IBM_2002].
```python
dir(object_name)
print [method for method in dir(object_name) if callable(getattr(object, method))]
```

## Tuples
Good python specific [walkthrough](http://openbookproject.net/thinkcs/python/english3e/tuples.html).

Group any number of items into a single compound value. Ordered, immutable, often used for data structures.

`x = (5,)` is a single element tuple

### Named Tuples - Quick Data Structs

```python
# namedtuple(typename, field_names, *, rename=False, defaults=None, module=None)
from collections import namedtuple
Point = namedtuple('Point', ['x','y'])
p = Point(11, y=22)
x,y = p #unpack
p.x + p.y #outputs 33

d = p._asdict() #returns ordered dict
Point(**d) #converts dict to namedtuple
p._replace(x=100)
```


# Syntactic Sugar

### Decorators
Basic idea, a decorator takes in a fn, adds some wrapper or functionality, and returns the same fn. So a decorator on `func_a` returns `func_a`, just with some additional functionality.
```python
def time_this(fn):
    def ret_fn(*args, **kwargs):
        import datetime
        before = datetime.datetime.now()
        x = fn(*args, **kwargs)
        after = datetime.datetime.now()
        print(f'Time Elapsed: {after-before}')
        return x
    return ret_fn

@time_this
func_a(input_1, input_2):
    do_something(input_1)
    do_something(input_2)
```

### Decorators w/ Inputs
Decorators can be further extended to take inputs, allowing a spread of functionality. E.g., permissioning (from the tut above):
```python
def requires_permission(sPermission):
    def decorator(fn):
        def decorated(*args, **kwargs):
            lPermission = some_fn_to_ret_permissions(sPermission)
            if sPermission in lPermission:
                ret_fn(*args, **kwargs)
            raise Exception('Permision Denied')
        return decorated
    return decorator

@requires_permission('admin'):
def delete_user(userid):
    usersdatabase.delete(userid)

@requires_permission('premium'):
def premium_checkpoint(savefile):
    savedatabase.save(savefile)
```

A meta, from the same tutorial
```python
def outer_decorator(*outer_args,**outer_kwargs):
    def decorator(fn):
        def decorated(*args,**kwargs):
            do_something(*outer_args,**outer_kwargs)
            return fn(*args,**kwargs)
        return decorated
    return decorator

@outer_decorator(1,2,3)
def foo(a,b,c):
    print a
    print b
    print c

foo()
```


# Appendix: Useful/Recurring FNs
_Resize Img fn_ (note: can kinda do `apply_tfms(size=some_number)` fn in fastai):
```python
# matplotlib.plot, Pillow as PIL, pathlib, os

def resize_one(fn, img, path, size):
    dest = path/fn.relative_to(path_land)
#     dest = path/fn.relative_to(path_og)
    print(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, size, use_min=True)
    img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
    img.save(dest, quality=90) #95 is most, 75 is default
```

##### Image plots
_asdf_:
```python
#magics are for displaying w/in ipynbs
%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.vision import *

# displays transformations of an image
def plots_of_one_image(img_url, tfms, rows=1, cols=3, width=15, height=5, **kwargs):
    img = open_image(img_url)
    [img.apply_tfms(tfms, **kwargs).show(ax=ax)
         for i,ax in enumerate(plt.subplots(
                               rows,cols,
                               figsize=(width,height)[1].flatten())]
```
_plot same img multiple times_ (demonstrate transforms):
```python
def _plot(i,j,ax): data.train_ds[0][0].show(ax, cmap='gray')
plot_multi(_plot, 3, 3, figsize=(8,8))
```

_Plot Multiple Imgs fn_:
```python
#matplotlib, numpy
def multi_plot(images:list, c:int=1):
  n_imgs = len(images)
  fig = plt.figure()
  for n, image in enumerate(images):
    a = fig.add_subplot(c, np.ceil(n_images/float(c)), n+1)
    if image.ndim == 2:
      plt.gray()
    plt.imshow(image)
  fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
  plt.show()

def _plot(i,j,ax): data.train_ds[0][0].show(ax, cmap='gray')
plot_multi(_plot, 3, 3, figsize=(8,8))
```

_image load, resize, and save fns_
```python
import torch
from PIL import Image

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std
```


[dive_into_python_ch4]: https://web.archive.org/web/20180901124519/http://www.diveintopython.net/power_of_introspection/index.html
[IBM_2002]: https://www.ibm.com/developerworks/library/l-pyint/index.html "might be old ;)"
