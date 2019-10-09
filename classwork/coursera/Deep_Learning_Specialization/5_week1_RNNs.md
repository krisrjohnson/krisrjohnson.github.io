
# RNNs

Activation at time t:
  `a<t> = g(Wa[a<t-1>, x<t>] + Ba)`
g typically tanh, might also do softmax for y_hat, Wa is the horizontal stacking of Waa, Wax

## GRU - Gated Recall Unit
Create a memory cell, `c<t>=a<t>` (for now), perform the typical activation from above, replacing `a<t>` with `c~<t>`.
Also perform a sigmoid on `a<t-1>`/`c<t-1>`, for the udpate gate. Sigmoids are basically 1 or 0, so update or don't update.
So the new `c<t>` becomes:
  `c~<t> = tanh(Wc[r_gate * c<t-1>, x<t>] + Bc)
  u_gate<t> = sigmoid(Wu[c<t-1>, x<t>] + Bu) // u<t> = vector of 1s or 0s
  r_gate<t> = sigmoid(Wr[c<t-1>, x<t>] + Br)
  c<t> = u_gate<t> * c~<t> + (1-u_gate<t>) * c<t-1>`


## LSTM
Has three gates instead of two, splitting apart the update gate into update and forget gates. Adds on an output gate. `a<t> != c<t>`

`c~<t> = tanh(Wc[a<t-1>, x<t>] + Bc)
u_gate<t> = sigmoid(Wu[a<t-1>, x<t>] + Bu)
f_gate<t> = sigmoid(Wf[a<t-1>, x<t>] + Bf)
o_gate<t> = sigmoid(Wo[a<t-1>, x<t>] + Bo)
c<t> = u_gate<t> * c~<t> + f_gate * c<t-1>
a<t> = o_gate * c<t>`


