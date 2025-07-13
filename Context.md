# Context

## Vocabulary Building

We are working on a machine learning theses. Our task is to reduce calculations required to compute BERT encoding by approximating it.
To achive this task we want to do the following:

1. [DONE] Build a vocabulary based on MS Marco dataset. This vocabulary should include n-grams of tokens and skip-grams of tokens based on their contextual meanings.
2. [DONE] Those n-grams should be built using statistics and semantics from the source dataset. We don't want to blindly enumerate all n-grams because that would end in too large vocabulary.
3. [Done] Then, for each n-gram and skip-gram within the vocabulary we need to calculate their BERT encoding and save it in an optimal way so that we can later efficiently reuse it.

## Model architecture and training

Then, we use this vocabulary and embeddings to train our own approximation model.
The goal of this model is to be a lightweight replacement for BERT.
The idea is that by using a lightweight neural network, we can approximate bert embedding outputs using pre-calculated embeddings of our vocabulary.
Here is the draft of the algoritm for this model:

1. We take some text as inputs, tokenize it.
2. We look for all the matches in our vocabulary of n-grams, skip-grams and regular tokens.
3. For each match, we take already pre-computed bert embedding that we created on previous step
4. We pass all found embeddings to the neural model that will generate the final embedding for the source text.
5. This final embedding must be as close to the BERT embedding of the source text as possible.

To make sure that the output embedding is as close as possible, we need to use original bert embeddings in the trainig process.
The draft for the training algoritm looks like this:

1. We take some source text, tokenize it and pass to BERT to get the embedding
2. We pass the source text through our own model.
3. We calculate scalar distance between our model's output vector and BERT's output vector, this will be our loss value
4. We adjust our lightweight model with backward propagation
