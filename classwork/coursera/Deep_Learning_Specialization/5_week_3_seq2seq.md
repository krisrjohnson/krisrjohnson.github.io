# sequence to sequence modeling

Encoding network -> decoding network to translate a sequence from one distribution to another, like literal translation or img captioning.

## Basic Models
translation: LSTM encoder of english sentences -> LSTM decoder into French
img captions: AlexNet sans softmax to encode image -> LSTM decoder to caption

## Picking the most likely sentence
Machine translation as building a conditional language model

The decoder is a language model that starts off with a_0 having a value instead of all 0's, hence conditional language model as a condition is fed through using a_0. So it's the probability of an English sentence conditioned on an input French sentence to translate from French to English, where the model will give the probability of different English translations. 

Obviously don't want to sample words, since you want recreatability/stability, rather an English sentence Y that maximizes the probability. Beam search > greedy search.

## Beam Search
Define a beam width, say 3. Take the 3 words with the highest probability as your first word (have to run through the full vocab). At the second layer, for each first word, calculate the probability of all words in the vocab, and take the 3 with the greatest conditional probability. 

So word 2 is P(y<1>|x) * P(y<2>|x, y<1>) for the 3 different y<1>'s. Could end up with (y_1=jane,y_2=is), (y_1=jane,y_2=visits), (y_1=in,y_2=september). Rinse and repeat.

## Refinements to Beam Search
Length Normalization - use sum of the log(P(y<\t> |x,y<1>,...,y<2>)) for t=1 to Ty, because the multiplication of probabilities (by definition < 1) with regular beam search can result in numerical underflow (rounded to 0). Log of a product is the sum of log, and summing is less prone to numerical underflow. 

Regular beam search prefers shorter length sentences, since it's a multiplication of probabilities << 1, so you can normalize/reduce the penalty from length by dividing by Ty. Aka, averaging the probabilities of all the conditionals. Furthermore, you can raise Ty to the _alpha_, a learnable parameter, as a heuristic/hack. No theoretical reason but works well in practice.

Run beam search then compute on the normalized log probability/normalized log likelihood object function to pick the best one. 

__Choosing beam width, B__: large B yields better result, but computationally more expensive. With vocab size V and output length Ty (unknown), computations is V + (B * V) * (Ty - 1)

Unlike BFS or DFS, Beam Search runs faster but not guaranteed to find exact maximum for arg max P(y|x)

## Error Analysis in Beam Search
If y_*, human provided translation, is 'Jane visits Africa in September' and y_hat is 'Jane visited Africa last September', was the RNN or the Beam Search?

Plug both through the RNN, and see which has the greater prob. If P(y_\*|x) > P(y_hat|x), then beam serach is at fault - it's not finding the highest probability answer! Otherwise the RNN is fault.

## Bleu Score
BiLingual Evaluation Understudy - Papineni et. al., Bleu: A method for auto eval of machine translation. Bigrams - pairs of words.

Given multiple sentences which correctly translate, 'the cat is on the mat' or 'there is a cat on the mat', 

Give each word in the Machine Translation (MT) credit for the max number of times it's in a human translation (HT). P_uniqgram is the sum of those clipped counts over total unigram count. P_ngram is similar, clipping MT to HT's n_gram (aka if 'the mat' is seen twice in the human translation, clip 'the mat' and 'mat the' to 2 total).

Plenty of open source Bleu score implementations. Provides a useful single number for evaluation.

## Attention Model Intuition
Bahdanau et. al., Neural MT by jointly learning to align and translate

For long sentences, Bleu score performance drops as NNs don't perform well. Humans translate long sentences a portion at a time, while encoder->decoder translations don't. Here's where Attention learning comes in.

Using bidirectional RNN, create weights from the RNN for each word - alpha<1,1>, alpha<1,2>,...,alpha<1,t> for the first word. These weights are the input to another RNN! These weights are how much attention we should pay to each word and output.

## Attention Model
a<0>, a<1>, a<1'>, a<2>, a<2'> means forward and back step (since the RNN is bidirectional) for a. Abbreviate to a<\t>. The context input to s<1> is the weighted attention weights (the a's sum to 1). 

c<1> = sum(a<1,t'>, a<\t'>) for all t'
aka, alpha<\t,t'> is amount of attention y<\t> should pay to a<\t'> (essentially a softmax).

Quadratic computation cost to run this algorithm, Tx * Ty computation cost.

Xu et al, neural image caption generation w/ visual attention

## Speech recognition - Audio data
audio clip, x -> transcript, y

Phoneme representations are no longer necessary due to much larger datasets.

CTC Cost for speech recognition, Tx = Ty on the order of 1000 inputs which is 10 secs of audio. The NN outputs equivalently 1k chars, but will collapse any repeated characters not separated by a special character. Example: the output '\_ttt__h_eeee_ \_' would collapse to the

## Trigger word detection system
Any device woken up by a trigger word, such as Siri or Alexa

RNN with training data with y's of when in an audio clip the trigger word is said, plus or minus a few seconds, otherwise 0's. The additional seconds of 1's are to reduce the sparseness of the 1's in the data.

That's it!