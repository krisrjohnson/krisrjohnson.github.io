
# word vectorization

## Word Embedding
Instead of working with one hot vectorization of the dictionary working with a vector of features per word in the vocab chosen, combined creates an embedding!

## Learning Word Embeddings - word2vec & GloVe
### Learning Word Embeddings
I want a glass of orange ___ -> [o4343, o9665, o1, o3852, o6163, o6257] = O
Multiply this vector of one hot word vectors by the word embedding matrix to create a vector of word specific embeddings: `O * E = [e4343, e9665, e1, e3852, e6163, e6257]`
Then take matrix of embeddings and connect to a fully connected layer and softmax activation to predict the next word. You might additionally just use a four word history, to make the fully connected layer easier. Need to train this fully connected layer.

context/target pairs, give the algo the last 4 words and the next 4 words and predict the word in the middle. Follows the same architecture as before, training a fully connected layer.

To train a word embedding you want bidirectional, close words for context. 

### Word2Vec

context to target errors, a previous word like orange can have multiple appropraite target words, juice,glass,my, etc for +/- 10 words around 'orange'.

__skipgram__:
Model, vocab size = 10k, want to learn a mapping from some context word, c, e.g. ("orange")\_6257 -> target, t e.g. ("juice")\_4834

O_c -> E -> e_c -> o_softmax -> y_hat. theta_t is the pramater associated w/ output t. 
Aka: Softmax: p(t|c) = e^(theta_t.T * e_c) / sum(e^(theta_j.T * e_c)); for j in all 10k words
L(y_hat, y) = -1 * sum(y_i * log(y_hat)); for i in all 10k words

Problems w/ softmax classification - have to use all 10k vocab words for the prediction step. Could use a hierarchical softmax, which does a binary tree to make computation size = log(n_Vocab). What is the classifier doing at each step? The tree is a starting node with two branches, for first half of the words and second half. In practice common words are brought closer to the starting node to speed up processing, while uncommon words are many branches deep. 

How to sample c? Sampling uniformly random from training corpus overepresents the,of,and,a etc. There are common heuristics to help remove lower the frequency of these high frequency words.

### Negative Sampling
Negative sampling speeds up the softmax objective which is similar to skipgram but more efficient.

Pick a context word and target word, to create a positive example. Then for k (5-20 for small, 2-5 for larger datasets) times, take the same context word and random words from teh dictionary and call those negative examples. It's okay if by chance one of the random words is actually in the original 10 word window of the context word. 

When we do training, instead of training binary classifiers on all 10k words in vocab, we'll train k+1, with the +1 for the actual target word now associated with y_hat = 1 (hence negative sampling). For sampling the k words, take a heuristic value assigning P(w_i) = freq(w_i) ^ .75 / sum(freq(w_j)^.75) for j in all 10k words. This lowers sampling of high freq words like the,and,a, etc while not overrepresenting incredibly uncome words like durian.

### GloVe Word Vectors
x_ij is how many times i appears in context of j (i.e. how many times is x within +/- say 5 words of j). x_ij = x_ji.

Minimize sum for i[1:10k], j[1:10k] of f(X_ij) times the square of (theta_i.T * e_j) + b_i + b'\_j - log(X_ij)
where f is a weighting fn, with basic rule f(X_ij) = 0 if x_ij = 0 (aka 0 * log(0) == 0))
theta_i and e_j are symmetric.


## Word Embedding Applications

### Sentiment Classification
Take a sentence, multiply each word by your now trained Embedding, and perform a function on the output encoded vectors, then push through a softmax for whatever you're trying to predict. In the case of 5 star rating predictions the softmax would output a length 5 vector. 

Can additionally push the embedding vectors of this sentence through an RNN with a single output layer, which is a softmax output of the 0-5 classes! This will generalize better than simple functions like adding the embeddings before the softmax layer.