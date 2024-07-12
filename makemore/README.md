### how this bigram model works (no NeuralNet)
bigram: using only the previous char to predict the next char. character-level prediction.
1. create a 2d list that has all english characters (and '.'). each element in the table is a certain character order.
2. normalize each row so that you can feed it in a sampler (torch.Generator and torch.multinomial). retrieve each character and store it into a list. 
3. loss: minimizing the average negative log likelihood (lower means the model is performing better).

### how this bigram model works (NeuralNet)
1. dataset: the previous character is fed into x and the next character is y. this structure is repeated across all the names.
2. weights are randomly initialized and inputs are one-hot encoded so they can be fed in the neural network.
3. probabilities calculated from matmul of input and weights.
4. Negative log likelihood loss is calculated using the true indices ys.

### MLP
1. the first layer looks at 3 previous words and embeds them in n dimensions using a lookup table.
2. each embedding set is concatenated and sent through a tanh layer.
3. the final layer is as long as the vocab size which holds the logits.
4. logits get softmax-ed into a probability distribution.

#### Calculating loss
prob[torch.arange(32), Y] - 
probs holds the probabilities of the next occurring character for each character in x. we use arange in the first index to see if the probability that that specific character appears next is close to the ground truth, determining the loss.

WaveNet -> RNN
* when growing the vanilla run, all we do is squash the input into a few neurons. this doesn't lead to better results. 
* 


### important concepts
* broadcasting semantics
* regularization loss -> tries to make weights to go zero which allow for uniform distribution. why do we want this?
* we know the model is underfitting when the validation/test loss is close to the test loss. this means the models needs to be larger.
* lower batch size means the loss is less accurate since we're doing gradient descent with a smaller sample size.
* batchnorm - makes a layer's distribution gaussian
* why use nonlinearities? if we didn't have any nonlinearities, all the layers would collapse into a single linear function. this limits a neural net's ability to approximate arbitrary functions, hence we include nonlinearities. 


### Links
https://colab.research.google.com/drive/1CJQeyOSecEJ8-lKK8g7QhP_ZEfWz_lQs?usp=sharing
https://colab.research.google.com/drive/187XiSKugmMAYVUM4-5R95_nEmCfhoOAc?usp=sharing