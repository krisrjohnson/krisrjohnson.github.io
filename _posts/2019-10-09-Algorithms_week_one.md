---
layout: single
tags: algorithms, coursera, computer science, java
categories: coursera
---

# Algorithms week 1

Dynamic Connectivity:
  Union - connect two objects.
  Find - is there a path connecting two objects?
  p is connected to p: reflexive
  if p is connected to q, q is connected to p: symmetric
  if p is connected to q, q connected to r, then p is connected to r: transitive


## Quick Find
  p and q are connected iff they have the same id, to union, change all components with who's id = p to be q
```python
id = list(range(8))
union (4,3) #id[4] now evals to 3
connected(3,4) #evals to True
union (3,2) #id[4],id[3] now eval to 2

class QuickFind():
  __init__(self, N):
    super(QuickFind, self).__init__()
    self.N = N
    self.id = []
    for i in range(self.N): id[i]=i

  def connected(p,q): return id[p]==id[q]
  def union(p,q):
    pid, qid = id[p], id[q] #have to extract this value as you'll overwrite id[p]!
    for i, _ in enumerate(self.id):
      self.id[i] = qid if self.id[i]==pid else id[i]
  #so each union call touches the entire array and union(N,N) takes N**2 time! O(N**2) if you will
```


## Quick Union
Same datastructure as before, but each entry contains a reference to its parent. And a root is where id[i]==i.
  init: `id = list(range(N))` O(N)
  root: `while(i!=id[i]): i = id[i]; return i`
  find: `root(p)==root(q)`, trees can get too tall, so find is O(N)
  union: `id[root(p)] = root(q)` O(N), again w/ tree height

int array id[] of length N


__Quick Union Improvements__:

## Weighted Quick Union
TL;DR: link root of smaller tree to root of larger tree to avoid huge trees

As an addition to regular quick find, keep track of the size of each tree. So a new int array sz[] of length N
When performing union(p,q): `if sz[root(p)] < sz[root(q)]: root(p) == root(q), else root(q)==root(p) #aka link smaller tree to root of bigger tree`
This makes maximum depth of any node log_2(N), since union(p,q) at least doubles the size of the larger tree but only increases the depth of the smaller tree by 1, and we initialize with id[i]==i. So if x is in the larger tree it's depth won't increase, if it's in the smaller tree it's depth increases by 1. You can only double the size of N numbers log_2(N) times.

So initialization is O(N), but union and find are O(log_2(N))


## Weighted Quick Find with Path Compression
Adding onto WQU, while performing root(p), set each node traversed to the final root value. _In practice_ actually set the root of i to its grandparents root: `while (i != id[i]): id[i] = id[id[i]]; i = id[i]` Instead of doing a two loop pass, which would require tracking all the nodes along the way of the root(), you do a single pass that partially flattens the tree as you go. Each subsequent root() call will further flatten the tree down.


## Union-Find Applications - Percolation
Percolation is a model for physical systems, it's an NxN grid where each site is open with probability _p_ or blocked with probability _1-p_

A system is percolated if the top and the bottom are connected by open grids, or sites. Aka, is there a path from the top to the bottom.

If _p_ is low, there won't be many open sites so likelihood of percolation is low. When N is large (it's an NxN grid), there is a sharp threshold of whether or not it'll percolate: __.593__! This was determined by doing Monte Carlo simulations using union find algorithms.


## Practice Quiz
1. Social Network connectivity
Given n members and log file containing m timestamps when friendships formed, design an algo to determine earliet time at which all members are connected, aka every member is a friend of a friend etc. Running time should be m log n or better.

A: Create separate list initialized as i = id[i], with length N. For every union, pop the id from the list for the smaller tree. When len(list)==1, everyone's connected.

2. Union-find, add a find() method so find(i) returns largest element in connected components containing i. E.g. find({1,2,6,9}) returns 9.

A: Assuming connected components means only the path up a tree, doesn't seem particularly clear. So need to add a step where unions cause higher componenets to rise to the top of tree, then find() becomes equivalent to root(). So during root(p), if id[p] < p, set id[id[p]] = p and id[p]=id[id[id[p]]], using an intermediate value and not updating i so that the check is performed again on the next loop.
So 9->4->2, id[9]=4, set id[id[9]]=9, id[9]=id[id[id[9]]], and i remains 9.

3. Successor with delete: n ints `S=list(range(n))`, def request(x) to be pop x from S and return smallest y in S where y>x. Takes log time or better.

A: Pop(x) should set id[x] to root(x+1), where that root() fn does the grandparent reset:`while (id[i] != i): id[i] = id[id[i]]; i = id[i]`. What happends when you pop the largest value though? Problem doesn't really explain that scenario, imagine you'd add error handling for where x+1 > N.
