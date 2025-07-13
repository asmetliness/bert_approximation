Perfect. I’ll now conduct deep research into model architectures suitable for approximating BERT embeddings under the following constraints:

* Input is a bag of matched n-/skip-gram vocabulary embeddings (order-invariant).
* Final output should approximate BERT’s CLS embedding.
* Longest-match vocabulary extraction is done prior to model input.
* Target task is dense retrieval (dot-product similarity).
* The goal is faster inference than BERT, with good quality retained.

I will look into lightweight attention-based pooling methods, gated additive encoders, and recent retrieval-optimized architectures. I’ll return with a full architecture proposal, training strategy, and a Python implementation sketch.


# Designing a Neural Network to Approximate BERT CLS Embeddings from Bag-of-Tokens

## Background and Problem Overview

Traditional BERT models produce powerful contextual embeddings (e.g. the `[CLS]` token embedding) that capture sentence meaning, but they are computationally heavy for long sequences. We aim to design a lightweight neural network that takes a **bag of pre-computed token embeddings** (from a custom vocabulary of n-grams and skip-grams) and **outputs an embedding close to BERT’s CLS embedding** for the entire input text. This approach leverages a pre-built vocabulary of meaningful n-grams/skip-grams (e.g. phrases) from a large dataset (MS MARCO), where each vocabulary token has an associated embedding (precomputed using BERT or a similar model). By matching the longest possible phrases in the text to this vocabulary, we obtain a set of token embeddings which serve as input to our model instead of the full token sequence. The goal is to drastically **reduce computation** (avoiding full BERT encoding for each query) while maintaining high semantic fidelity in the output embedding. This embedding will be used for *dense retrieval* (e.g. via cosine or dot-product similarity), so it must preserve meaningful similarities between texts.

Key requirements for the model are:

* **Order-invariant input processing:** The model should treat the input as a *set* of token embeddings (since we use a bag of n-gram/skip-gram tokens). The presence of particular tokens (phrases) should matter more than their order, as the input is obtained via longest-match tokenization.
* **High semantic accuracy:** The output vector should approximate the original BERT `[CLS]` embedding as closely as possible in meaning, ensuring that similar texts have similar embeddings.
* **Fast inference:** The network should be much faster than BERT (ideally *O*(n) or even *O*(1) in input length, as opposed to Transformers’ *O*(n²) attention), enabling quick embedding of sequences up to 512 tokens.
* **Handling long inputs:** It must handle up to 512 tokens worth of text (which after phrase matching could be fewer set elements) without degradation. Using *longest-match token selection* means many words may be grouped into one token, reducing the number of input elements. The model should be robust to variable set sizes and not overly sensitive to large counts of tokens.
* **Use of precomputed embeddings:** The input token embeddings are obtained beforehand (e.g. by running BERT or a similar model on the token/phrase itself and storing it). The network does not need to encode raw text, only to **aggregate these embeddings**. This shifts most of the heavy semantic encoding work to an offline step, allowing online queries to be fast.

With this setup in mind, we can frame our model as a *set encoder*: it receives a set (or multiset) of vector representations and must output a single vector. We will explore possible architectures that meet these criteria and then propose a final design.

## Candidate Architectures for Bag-of-Embeddings Input

Designing a network on an unordered set of token embeddings requires permutation-invariant architectures. Several applicable frameworks and pooling mechanisms can be considered:

* **DeepSets (Sum/Mean Pooling + MLP):** Zaheer et al. (2017) showed that any permutation-invariant function can be decomposed into transformations on individual elements followed by a symmetric pooling (like sum). In a DeepSets model, each token embedding **eᵢ** is first transformed by a function φ (e.g. a small neural network) and then all transformed vectors are summed or averaged; a final function ρ produces the output. This architecture naturally handles variable set size and is order-agnostic. It’s also computationally efficient (linear in number of tokens). However, simple averaging or summation treats all tokens equally, which may not fully capture which phrases are most important. We likely need to **learn to weight or select** tokens, which leads to attention/gating variants. Still, DeepSets provides a strong foundation and is a universal approximator over sets.

* **Deep Averaging Network (DAN):** This is a specific instance of the DeepSets idea used in the Universal Sentence Encoder (Cer et al. 2018) and by Iyyer et al. (2015). In a DAN, one simply averages the embeddings of words (and possibly n-grams) and then passes the averaged vector through a few feed-forward layers. It forgoes any complex sequence modeling, trading a bit of accuracy for speed. **Computational cost is constant with respect to input length** (apart from the initial averaging), making it extremely fast. In our case, since our tokens include n-grams and skip-grams, averaging them is similar to what DAN does (where bigrams were included to inject some order information). The downside is that pure averaging is insensitive to word order (beyond what n-grams capture) and assumes each token contributes equally. Nonetheless, a DAN provides a useful baseline architecture.

  &#x20;*Example of a Deep Averaging Network architecture.* In this approach, token embeddings (for words and bigrams, e.g. “hello”, “world”, “hello world”) are **averaged** into a single vector, which is then passed through multiple feed-forward layers to produce the final sentence embedding. This method is **order-invariant** and highly efficient, although it may lose some nuance of word order. We can draw inspiration from DANs for their speed and simplicity, while planning to introduce mechanisms to weight more important tokens.

* **Gated/Attentive Pooling:** To improve on simple averaging, we can introduce a learned **attention mechanism or gating function** that assigns different weights to each token’s embedding before combining them. For example, one can compute an importance score *αᵢ* for each token embedding eᵢ (via a small neural network or even a single linear layer + nonlinearity), and then take a weighted sum of the token representations: **v = ∑ᵢ αᵢ · φ(eᵢ)**. This allows the model to **focus on important parts of the sentence** – an idea backed by prior work on self-attentive sentence embeddings. Gated attention was used by Chen et al. (2017) for sentence encoding, where a gating function learned which words to emphasize based on their information content. Unlike full self-attention, this approach is lightweight: it’s essentially a one-layer attention pooling. We could implement this by, for instance, feeding each eᵢ through a small MLP to get a scalar gate or attention score, and normalizing these scores (softmax) to weights. The result is a **learned weighted average** where the model can upweight key phrase embeddings (e.g. maybe a rare bigram that determines the sentence meaning) and downweight less informative ones. Gated pooling still treats tokens independently (aside from competing for higher weight), but it adds a useful **differentiable selection** mechanism without heavy cost.

* **Lightweight Self-Attention (Set Transformer):** Another extension is to allow tokens to interact with each other before pooling. The *Set Transformer* (Lee et al. 2019) introduced attention blocks that model pairwise interactions among set elements while remaining permutation-invariant (by omitting positional encoding). It also proposed a **Pooling by Multi-Head Attention (PMA)** mechanism, where a small number of learnable “seed” vectors attend to all elements to produce a fixed-size output. In our context, one could imagine a single “CLS-seed” vector attending to the token embeddings to directly compute the output embedding. This would be akin to adding one Transformer layer where the query is a global embedding and keys/values are our token embeddings. Such an approach can capture interactions (e.g. two tokens together indicating a different meaning than individually) and complex weighting. However, introducing self-attention increases complexity and may hurt inference speed, especially if the set of tokens is large. A single attention layer is *O(n)* in our case (with a fixed number of queries like k=1 or small k), but full token-to-token attention would be *O(n²)* which we want to avoid for large n. For a **“lightweight” attention**, we would likely restrict to *one or two attention heads* and minimal layers, just enough to capture basic token interactions or importance signals. This can be seen as a compromise between a full Transformer and a simple weighted sum.

In summary, we have a spectrum: at one end, **simple pooling (mean/sum)** which is fastest but least expressive; in the middle, **attentive pooling/gating** which weights tokens individually; and at the other end, **a shallow attention network** which allows some token interaction and dynamic weighting. Considering our needs, we favor simplicity and speed, but we also need accuracy – the model should learn to mimic BERT’s complex encoding. Empirically, incorporating an attention mechanism to highlight important tokens can significantly improve embedding quality for tasks like retrieval, with only a minor speed cost. Therefore, we lean towards a DeepSets-style model *augmented with a gating/attention pooling* as our final design. This keeps the architecture simple (few feed-forward layers, one pooling operation) but sufficiently flexible to re-weight token contributions based on context.

## Proposed Model Architecture

Taking the above into account, we propose a **hybrid DeepSets + Attention pooling model** to approximate BERT’s CLS embeddings. The model has the following structure:

1. **Input layer:** A set of input token embeddings **{e₁, e₂, ..., e\_m}**, where each eᵢ is a d-dimensional vector (e.g. d=768 if using BERT-base embeddings, or possibly a lower dimension if using a distilled model). The tokens correspond to longest-match n-grams or skip-grams found in the text. We do not assume any particular ordering of eᵢ; the set is treated as unordered. (If the implementation needs a fixed shape, the set can be given as a padded list and a mask, but the model itself will not use positional information.)

2. **Token transformation (φ):** Each embedding eᵢ is passed through a small feed-forward network to produce a **hidden representation hᵢ**. For example, we might use a two-layer perceptron:
   $hᵢ = \text{ReLU}(W_1 eᵢ + b_1)$
   $hᵢ = W_2 hᵢ + b_2,$
   so that hᵢ has some desired hidden dimension H. This transformation φ can project the raw embeddings into a space more suited for combination, and potentially increase the capacity of the model. (In practice, setting H equal to the target embedding size – e.g. 768 – works, or we could use a slightly higher H for more capacity and then map down to 768 later.) Each token’s vector is transformed *independently* by the same φ, which means this step is parallelizable across tokens.

3. **Attention-based pooling:** We introduce a *gating network* that computes an importance weight αᵢ for each token. One straightforward way is:

   * Compute a scalar score sᵢ for each token via a linear layer on the original embedding or the hidden hᵢ. For instance:
     $sᵢ = w^\top (\tanh(W_g eᵢ + b_g)),$
     where W\_g and w are learned parameters (W\_g projects eᵢ to some smaller vector, and w^T produces a scalar). This is analogous to the attention score computation in many models (a one-layer MLP attention). We could also use hᵢ instead of eᵢ here, or even a combination, but using the original eᵢ for scoring keeps the paths separate (one path to compute content, one to compute weight).
   * Apply a softmax over these scores across all tokens in the set:
     $\alphaᵢ = \frac{\exp(sᵢ)}{\sum_{j=1}^{m} \exp(s_j)}.$
     This yields a *normalized attention weight* for each token such that ∑ᵢ αᵢ = 1. (An alternative is to use a sigmoid gating and then normalize by sum or just use as weights, but softmax ensures a normalized convex combination, which often works well for focusing on a few key tokens.)
   * Now compute a weighted aggregate of the token representations:
     $z = \sum_{i=1}^{m} \alphaᵢ \cdot hᵢ.$
     This **attentive pooling** means the model can assign higher α to tokens that are critical to the sentence meaning and lower α to less important ones. For example, if the input text is *"Obama visited France"*, and suppose the vocabulary matched “Obama” and “France” as tokens, the model might learn to give those more weight than common verbs like “visited” (if “visited” is a token too). The gating network effectively learns to mimic BERT’s behavior of focusing on certain tokens for the CLS output.

   *Rationale:* This attention pooling adds minimal overhead – one linear layer for scoring and a softmax – but allows the network to be sensitive to which token (phrase) embeddings matter more in a given context, rather than just averaging everything. It’s a form of content-based weighting, inspired by self-attention mechanisms for sentence embeddings. If certain combinations of tokens are semantically salient (e.g. presence of a specific bigram), the model can reflect that by increasing its weight. This addresses one weakness of simple averaging, where e.g. adding a negation word might only slightly change the average; with attention, a “not” token could be learned to have a high weight (in magnitude or maybe negative effect via the MLP) to properly invert meaning.

4. **Output transformation (ρ):** The pooled vector *z* is passed through another feed-forward transformation to produce the final output embedding **y** (the BERT-approximate CLS vector). For instance:
   $u = \text{ReLU}(W_3 z + b_3),$
   $y = W_4 u + b_4.$
   In many cases, we could even make ρ just an identity if we set things up such that z is already the correct dimension. However, having an extra linear layer or two as ρ helps the model adjust the combined features of all tokens to better match BERT’s output distribution. It can correct any biases from the pooling step. Essentially, φ handles per-token feature extraction, the attention weights α handle selection, and ρ mixes the aggregated information into the final embedding. The overall network can be seen as a kind of two-stage MLP around a pooling operation (which is a common template for set functions).

**Parameter counts and complexity:** This model is extremely lightweight compared to BERT. If d = 768 and we use hidden size H = 768, the φ and ρ networks are on the order of a few hundred thousand parameters each (for example, W1 and W2 are 768x768 matrices, etc.), totaling perhaps a few million parameters at most. The attention scoring network is similarly small. In contrast, BERT-base has 110M parameters. Inference complexity is dominated by matrix multiplications for φ and ρ (linear in m and d), plus the softmax over m tokens. If m is, say, on average 50 or 100 (depending on how many phrases an average 512-word text yields), this is extremely fast. Even in worst-case m=512 (no phrases combined), we have on the order of 512\*768 multiplications per layer, which is \~393K ops per layer – negligible compared to BERT’s multi-head attention in 12 layers over 512 tokens. Thus, **inference is essentially O(m \* d \* H)** for the feed-forward operations, which is much faster than BERT’s self-attention *O(n²)* scaling for long sequences. In practice, this model should be **tens of times faster** than BERT, if not more, for 512-token inputs (BERT needs multiple transformer layers, each with 512-length attention). It’s also very memory-efficient.

**Handling 512-token sequences:** Our model itself has no sequence length limit since it operates on sets. However, the input pipeline must handle splitting text into vocabulary tokens. The use of longest-match phrases ensures we take the largest meaningful chunks, which keeps *m* (the number of tokens) manageable. For extremely long inputs, the main load is computing attention scores for each token and summing, which is linear. If needed, one could chunk very large sets or limit the number of tokens by dropping very low-weight tokens (though that would require a preliminary pass to estimate α). Generally, this model will comfortably handle up to 512 original text tokens (resulting in m ≤ 512, likely much less on average).

**Order and overlap considerations:** Because we ignore token order, any information about word sequence beyond the scope of the chosen phrases is lost. This is a design decision to maximize speed. The hope is that by including n-grams (and skip-grams) in the vocabulary, a lot of local ordering is implicitly captured (e.g. the phrase “credit card” as one token vs. “card credit” which likely wouldn’t appear). Global reordering that doesn’t change which n-grams appear will indeed produce the same bag – a known limitation (the model might encode "dog bites man" and "man bites dog" similarly if the same bigrams or unigrams are present). In practice, the difference in meaning often comes with different combinations of tokens. If order-sensitive distinctions are critical, we might need to incorporate some positional encoding or sequential modeling, but that would complicate the model and slow it down. We accept some loss of nuance for speed, noting that the teacher model’s embeddings themselves (especially a model fine-tuned for retrieval) might not heavily encode exact word order for tasks like semantic search. Our approach is similar in spirit to the Deep Averaging Network which also ignores order but was still effective for many tasks, especially when trained on a broad semantic objective.

To summarize the architecture: *each token embedding passes through a small MLP; an attention mechanism then weighs and sums these token representations; finally another MLP produces the output*. This design is **permutation-invariant**, lightweight, and should approximate BERT’s CLS embedding function. Next, we discuss how to train this network to mimic BERT.

## Training Procedure and Optimization

Training the model will involve a form of **knowledge distillation** from BERT (or a BERT-like teacher model). The dataset for training can be the same large corpus from which the vocabulary was built (e.g. MS MARCO passages, or any text where we can obtain BERT embeddings). We do not need labeled data for an external task; instead, the “label” for each input text will be the BERT-produced embedding for that text.

**Training data preparation:** For each training example (a text sequence up to 512 tokens):

1. Compute the *teacher embedding* by running the text through the **BERT model** (the one we want to approximate). This could be BERT-base or a distilled version fine-tuned for retrieval. In the context of MS MARCO, one might use a model like `msmarco-distilbert-dot-v5` (a DistilBERT fine-tuned for passage ranking) as the teacher, since the vocabulary embeddings were also derived from it. Ensure we obtain a single vector representation – typically the `[CLS]` token output or a pooled output. Let’s call this vector **t** (dimension 768).
2. Tokenize the text using our **hybrid vocabulary tokenizer** (likely a trie-based longest-match tokenizer that finds the n-grams/skip-grams). This yields a set of tokens (phrases) that cover the text. For each token, look up its precomputed embedding (these were generated in a prior offline step by encoding the token itself with BERT). Collect these embeddings into a list. This list (of vectors **e₁...e\_m**) is the input to our model.
3. Feed the set of embeddings into our model (which performs the φ transform, attention pooling, ρ transform) to produce an output vector **y**.
4. Compute a **loss** between the output y and the teacher’s vector t. A suitable loss for embedding regression is **Mean Squared Error (MSE)**, i.e. \$\mathcal{L} = |y - t|^2\$. We can also use **cosine distance** (1 - cos similarity) as the loss to focus on directional alignment if we care primarily about cosine similarity match. Another option is **Mean Absolute Error (L1)**, but MSE will heavily penalize larger errors in any dimension, which might be useful to get the vector precisely right. In practice, MSE on normalized vectors or a combination of MSE and cosine loss can be effective in training student embeddings to align with teacher embeddings. For simplicity, we’ll consider MSE on the raw embeddings (the model can learn to scale appropriately).
5. Backpropagate the loss to update the model parameters (the weights in φ, ρ, and attention networks). We **do not update or backprop into BERT** – the teacher is fixed. We also typically keep the input token embeddings fixed (they were precomputed by BERT and represent our “input features”). Freezing the token embeddings reduces the number of parameters drastically and ensures the input semantics remain those of BERT. However, an alternative is to allow a little fine-tuning of those embeddings during training to better align with the student model’s needs (this would be akin to learning a better basis for the student). Given the vocabulary might be large (tens of thousands of tokens), it’s usually best to keep them fixed for faster training and to avoid overfitting small variations. Our model will learn to re-weight and combine them optimally.

**Batching strategy:** We will train with batches of examples for efficiency. Preparing a batch involves:

* Collating multiple texts’ token sets. Since each text can yield a different number of tokens, we need to **pad** or otherwise handle varying set sizes. A simple method is to pad the list of token embeddings for each example to the maximum set size in the batch with a special zero vector and use a **mask** to indicate which entries are real. Our model’s attention mechanism can ignore padded entries by masking them out in the softmax (set their score to a large negative number before softmax so they get nearly zero weight). Alternatively, we can sort examples by set length and bucket them, or use dynamic batching where each batch contains examples of similar token counts, minimizing padding. Given that 512 token texts might yield at most 512 single-word tokens (worst case), padding to that isn’t too bad if batch size is not enormous. But in practice, longest-match will reduce that, and we can choose a batch size that balances memory and compute.
* We feed the batch through BERT to get all teacher embeddings in one go (if using a GPU and the batch size isn’t too large, this is efficient thanks to BERT’s optimized batching). Similarly, we feed the batch of token sets through our model (which can also be vectorized since it’s basically matrix ops on padded tensors).
* The **batch loss** can be the average of per-example MSE losses. Using a vectorized implementation (PyTorch, etc.) this comes naturally.
* It’s important to shuffle the training data and possibly precompute the tokenized forms to avoid doing the vocabulary matching on the fly (though that could be done on CPU in parallel).

**Optimization:** We will use a standard optimizer like **Adam** (adaptive moment estimation) which works well for knowledge distillation tasks. A typical learning rate might be 1e-3 for our small network (since it’s not as deep as BERT, we can afford a slightly higher LR than used in fine-tuning BERT, which is usually 2e-5 or 3e-5). We should monitor validation loss (maybe on a held-out set of texts and their BERT embeddings) to avoid overfitting. Early stopping or learning rate decay can be used if we see the loss plateau. Because our output is a continuous vector, there’s no single “accuracy” metric, but we can track the **Cosine similarity** between y and t on the validation set or the **MSE** directly as a measure of performance. High cosine (close to 1) or low MSE indicates the student is matching the teacher well.

**Loss function details:** If using MSE, it might help to **normalize both y and t** to unit length (if the retrieval will ultimately use cosine similarity). This ensures the model focuses on getting the direction right, not the norm. However, if the teacher embeddings have meaningful norm differences (e.g. maybe BERT encodes some confidence in the norm), we might instead train on the raw vectors. We could also combine objectives: e.g. minimize MSE + λ\*(1 - cos(y,t)) to explicitly encourage high cosine alignment. Since dense retrieval often cares about cosine similarity, high cosine alignment is desirable. Another training approach is to perform *contrastive learning*: use pairs of texts (like in Sentence-BERT training) and train the student to produce higher similarity for real pairs vs random pairs, *matching the teacher’s similarity scores*. However, that would require sampling negatives and is more complex; given we have the teacher’s actual vectors, a simpler direct regression is straightforward and usually effective.

**Batch size considerations:** BERT is the bottleneck during training. If using a smaller teacher like DistilBERT, we can afford a larger batch (maybe 32 or 64 examples per GPU). If using BERT-base as teacher, memory might limit batch size to e.g. 16. We can also precompute and store teacher embeddings for the entire training corpus to avoid running BERT repeatedly. This speeds up training substantially (at the cost of disk space for storing all those vectors). Since our task is offline training, this precomputation is viable: we already computed embeddings for vocabulary tokens, so computing embeddings for full texts is analogous. If we precompute, then each training step just loads the teacher vector from memory instead of running BERT. This was likely considered in the project to speed up the training loop.

**Training duration:** Because the model is small, it will train much faster per step than BERT. The main time consumption is potentially the BERT inference for teacher (if not precomputed). With precomputed teacher vectors, training becomes a simple feed-forward and MSE calculation, which could allow using large amounts of data (millions of text examples) to really tune the student. This is often necessary for knowledge distillation to approximate the teacher on all sorts of inputs.

**Regularization:** We can use techniques like dropout within φ or ρ networks if overfitting is a concern, but since our input comes from a rich distribution of texts, overfitting is less likely than in small supervised datasets. Weight decay (L2 regularization) on the parameters (except perhaps bias terms) can help keep the model stable (a small weight decay like 1e-5 or 1e-6 with Adam is common).

After training, we should evaluate the model on some semantic similarity or retrieval tasks to ensure the embeddings perform well. If the cosine similarity between student and teacher embeddings on a random set of texts is very high (say >0.95 on average), we’ve successfully approximated the teacher. The true test is in downstream retrieval: replacing BERT with our model and seeing if retrieval quality stays high (perhaps only a slight drop).

We will now provide a simplified **PyTorch implementation** of the described model and a sample training loop to illustrate how it all comes together.

## Implementation: PyTorch Model and Training Loop

Below is a Python implementation using PyTorch. We define the model class with the architecture described, and then show a pseudo-training loop. This code assumes that we have access to a function `get_token_embeddings(text)` that returns the list of precomputed token vectors for a given input text (using the vocabulary matching), and a function `get_bert_embedding(text)` that returns the teacher BERT CLS embedding for the text. In practice, `get_bert_embedding` might be replaced by looking up a stored vector or by running a transformer model. We also assume we have a DataLoader providing batches of texts.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class BERTApproximator(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, output_dim: int):
        """
        embed_dim: dimension of input token embeddings (d)
        hidden_dim: dimension of intermediate hidden layer (H)
        output_dim: dimension of output embedding (e.g. 768)
        """
        super(BERTApproximator, self).__init__()
        # Token transformation φ: two-layer feed-forward
        self.phi = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)  # you can include a nonlinearity here or omit if not needed
        )
        # Attention score network: one hidden layer then scalar score
        self.attn_score = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),   # compute features from original token embedding
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)            # scalar score for each token
        )
        # Output transformation ρ: map pooled hidden to output dim
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
            # We could add a nonlinearity or extra layer here if needed
        )
    
    def forward(self, token_embeds: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        token_embeds: Tensor of shape (batch_size, max_tokens, embed_dim)
        mask: Tensor of shape (batch_size, max_tokens) with 1 for real tokens and 0 for padding.
        """
        # 1. Transform each token embedding
        # Shape: (batch, max_tokens, hidden_dim)
        h = self.phi(token_embeds)
        
        # 2. Compute attention scores for each token
        # It's possible to use h for scoring, but here using original embed for scoring as defined
        scores = self.attn_score(token_embeds).squeeze(-1)  # shape: (batch, max_tokens)
        
        # 3. Apply mask to scores: set very negative where mask is 0 (pad positions)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. Normalize into weights
        weights = torch.softmax(scores, dim=1)  # shape: (batch, max_tokens)
        # (Using dim=1 assuming max_tokens is the second dimension)
        
        # 5. Weighted sum of h vectors
        # Expand weights for multiplication
        weights_expanded = weights.unsqueeze(-1)  # shape: (batch, max_tokens, 1)
        pooled = (h * weights_expanded).sum(dim=1)  # shape: (batch, hidden_dim)
        
        # 6. Output transformation
        output = self.rho(pooled)  # shape: (batch, output_dim)
        return output

# Example usage:
embed_dim = 768   # dimension of token embeddings
hidden_dim = 768  # hidden dimension (can tune this)
output_dim = 768  # output embedding dimension (match BERT CLS size)

model = BERTApproximator(embed_dim, hidden_dim, output_dim)
model.train()

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Dummy example batch (for illustration; in practice get real data)
texts = ["Sample query text 1", "Another example sentence"]  # batch of 2 texts
# Suppose get_token_embeddings returns a list of arrays (one per text)
batch_token_embeds = [get_token_embeddings(tx) for tx in texts]  
# Let's pad them to same length
max_len = max(len(tok_list) for tok_list in batch_token_embeds)
mask = []
padded_embeds = []
for tok_list in batch_token_embeds:
    # tok_list is a list/array of shape (n_tokens, embed_dim)
    n = len(tok_list)
    mask.append([1]*n + [0]*(max_len - n))
    if n < max_len:
        # pad with zeros
        pad_vecs = [torch.zeros(embed_dim) for _ in range(max_len - n)]
        tok_list = torch.vstack([torch.tensor(tok_list), torch.stack(pad_vecs)])
    else:
        tok_list = torch.tensor(tok_list)
    padded_embeds.append(tok_list)
padded_embeds = torch.stack(padded_embeds)    # shape: (batch, max_len, embed_dim)
mask = torch.tensor(mask)                     # shape: (batch, max_len)

# Forward pass
pred = model(padded_embeds, mask)  # shape: (batch, output_dim)
# Get teacher embeddings for each text in the batch
with torch.no_grad():
    teacher_vecs = [get_bert_embedding(tx) for tx in texts]
teacher_vecs = torch.tensor(teacher_vecs)     # shape: (batch, output_dim)

loss = criterion(pred, teacher_vecs)
loss.backward()
optimizer.step()
```

**Explanation:** In the model above, `self.phi` implements the token-wise feed-forward network φ. We used two linear layers with ReLU; you could simplify to one linear if you find it sufficient, or add another for more depth. `self.attn_score` computes the attention score sᵢ for each token (we used the original embedding as input here). We mask out padding tokens by setting their score to -1e9 (a standard trick so that after softmax they get nearly zero weight). We then apply `softmax` to get αᵢ weights. The hidden representations `h` are combined by the weights to get a pooled vector, and `self.rho` produces the final output of dimension 768.

We kept things fairly simple (e.g., no activation after the final linear – since we are regressing to a teacher vector, a linear output layer is fine). If we wanted, we could add a non-linearity in `self.rho` or even use a residual connection (like add `pooled` to output if dimensions match) to help optimization, but it might not be necessary.

The training loop snippet creates a batch of two texts, gets their token embeddings, pads them, and feeds through the model. In practice, one would wrap this in an actual loop over many batches and epochs, and integrate with PyTorch’s DataLoader for efficiency.

**Batch mask handling:** We ensured padded entries don’t contribute by masking. Another approach is to pack the sequences (since order doesn’t matter, we could just concatenate all token embeddings in the batch and somehow aggregate, but that complicates mapping outputs to each example). Using a mask is straightforward.

**Memory note:** If a text has *m* tokens, φ will produce m hidden vectors. For very large m, this might be memory heavy in a large batch. However, 512 max tokens \* 768 dims \* batch\_size (say 32) is about 32*512*768 \~ 12.6 million floats, which is around 50MB – not trivial but manageable on a modern GPU. Usually *m* will be much smaller due to phrases.

**Potential improvements:** We could consider multi-head attention pooling (e.g., produce multiple sets of weights focusing on different aspects, akin to Lin et al.’s multi-head sentence attention). This would mean output = \[y₁; y₂; ...; y\_k] combined or averaged, etc., and maybe yield a richer embedding. But it would also increase output size or require another projection. Given retrieval embeddings are typically one vector, we stick to one pooled vector. If we wanted the effect of multi-head (capturing multiple semantic facets), we could set k=2 or 3 seed vectors in a PMA style and then average them or concatenate and project. This is an advanced tweak and would add minor complexity and parameters; we assume the single-head attention is sufficient when trained on a large corpus.

## Inference Speed and Additional Optimizations

Once the model is trained, using it in a production retrieval pipeline offers several advantages and some possible optimizations:

* **CPU/GPU Deployment:** The model is small enough to run on CPU with high throughput, especially if using optimized linear algebra libraries. Batch inference for multiple queries can further utilize vectorization. On GPU, the overhead of launching the model is minimal compared to BERT, so real-time embedding of queries or documents will be much faster (orders of magnitude faster for long documents).

* **Lower precision:** We can exploit lower numerical precision to speed up inference. After training, the weights can be quantized to 8-bit integers (INT8) or 16-bit floats. Libraries like PyTorch or TensorRT allow quantization with little loss in embedding quality. Since our network is straightforward matrix ops, it quantizes well. 8-bit inference could *double* throughput or better, and it reduces memory usage by 4x, which is beneficial if deploying on CPU at scale.

* **Dimensionality reduction:** The BERT CLS embeddings are 768-D. For retrieval, it might be sufficient to use a lower-dimensional vector (e.g. 256 or 300) to save space and computation in the similarity search. We could modify the output\_dim to a smaller size and train the model accordingly (perhaps adding a linear projection on the teacher side as well during training). This introduces a quality trade-off (slightly lower recall/precision in retrieval) for a gain in speed and storage. Even if we keep 768 for training fidelity, we could apply **PCA or autoencoder compression** on the output embeddings after training, or directly train with a smaller output by adding a layer that distills 768 -> 256 while training (this would effectively learn a compressed embedding with minimal loss). Many real-world systems compress embeddings for faster ANN search.

* **Approximate Nearest Neighbor (ANN) search:** While not part of the model itself, it’s worth noting that after obtaining embeddings, one typically uses ANN indices (like Faiss) for large-scale retrieval. Our model’s speed helps generate embeddings quickly; using cosine similarity with an optimized library will make the end-to-end retrieval very fast. If using cosine, ensure to L2-normalize the output embeddings (either within the model or as a post-processing step) so that dot product in ANN corresponds to cosine. This normalization can be built into the model as well (e.g., a final `output = output / ||output||` operation for inference).

* **Caching and reuse:** Because the model input is a set of static token embeddings, one could cache intermediate results for very common tokens. For example, if a document is seen often, its token set and even the model’s intermediate pooled result could be cached. However, since the model is so fast, caching at the document level may be unnecessary except in extreme throughput scenarios.

* **Pruning tokens by importance:** During inference, we could potentially drop some tokens to speed up the computation if we know they carry little weight. For instance, if the vocabulary includes many common uninformative tokens (like stopword unigrams which might slip through), the attention mechanism *should* learn to give them low weight. If we wanted, we could pre-filter input tokens by ignoring those that are extremely common or have IDF below a threshold, etc. This would reduce m slightly. But this is usually not needed as the computational cost of even 500 tokens is tiny for our model, and we risk losing information if we drop tokens arbitrarily. The learned attention is a safer way to handle it.

* **Parallelizing over tokens:** Our model operations (the φ transformation and score computation) are embarrassingly parallel over the token dimension. On GPU, this will be done in parallel naturally as matrix ops. On CPU, one can use vectorized NumPy/PyTorch or multi-threading to compute these in parallel. Ensuring the matrix multiply libraries are using multiple cores will give near-linear speedup with number of cores for large m. Since m might not be huge, this might not matter, but if embedding a batch of long documents on CPU, multi-core utilization is helpful.

* **Model compression:** If further speed is needed with slight accuracy trade-off, one could consider **knowledge distillation at the architecture level**: e.g., train an even simpler model to mimic this model’s outputs (though that starts getting meta). Or prune the hidden dimension (if we find 768 hidden is overkill, we could try 384 or 256 hidden neurons – that would drastically cut down parameters and multiplications). Starting with a smaller hidden size in φ and ρ is a straightforward way to lighten the model if profiling shows bottlenecks. Given the teacher’s embeddings live in a 768-D space, using a much smaller hidden might limit accuracy, so one might gradually reduce and see how it affects results.

* **Quality trade-offs:** If the model isn’t achieving the desired accuracy, consider incorporating more of BERT’s knowledge. Some ideas: (1) Use a small *self-attention block* among the token embeddings before pooling (for example, a single Transformer encoder layer without positions). This would let tokens influence each other (maybe capturing negation or phrase composition more explicitly). It adds some cost but maybe still acceptable. (2) Use multi-head attention pooling (as mentioned, multiple attention heads) to capture different aspects of meaning. (3) Fine-tune the vocabulary embeddings themselves: after some training epochs, you could unfreeze the token embedding vectors and allow them to adjust slightly to better fit the student model’s needs. This essentially learns a new embedding space that’s still initialized from BERT’s, which might improve approximation at the cost of diverging from pure BERT values. It’s a form of *dictionary fine-tuning*. One must be careful to regularize or restrict this (so embeddings don’t drift too far, undoing the advantage of good initialization). A possible approach is to tie the embedding adjustments to their BERT original by adding a regularization term that penalizes the student’s token embeddings from moving too far from the initial BERT-based embeddings.

In conclusion, the designed model (DeepSets with attention pooling) provides a viable solution to approximate BERT’s CLS embedding quickly. It leverages the strength of a carefully built n-gram/skip-gram vocabulary and offloads the heavy semantic encoding to an offline step (computing token embeddings). Our network then **learns to compose those embeddings** into a sentence representation that closely matches BERT’s output. By training on a large corpus with a direct regression loss, the model can achieve high fidelity to BERT (often a small angular error in embedding space) while being efficient at query time. This approach is especially suited for dense retrieval, where embedding many texts quickly is crucial. We have outlined the architecture, provided an implementation, and suggested training and inference optimizations. With this in place, we expect to attain BERT-like retrieval performance with a fraction of the computation, enabling deployment in latency-critical or high-throughput environments.

**Sources:**

* Zaheer et al. *Deep Sets*, NeurIPS 2017 – introduced permutation-invariant networks using sum pooling.
* Cer et al. *Universal Sentence Encoder*, 2018 – used a Deep Averaging Network (DAN) with word and bigram embeddings for quick sentence embeddings.
* Chen et al. 2017 – demonstrated gated self-attention pooling to focus on important words in sentence encoders.
* Lee et al. *Set Transformer*, ICML 2019 – proposed attention-based set encoding and pooling by multi-head attention.
* *User-provided context (MS MARCO vocabulary)* – outlined building a vocabulary of n-grams/skip-grams and using their BERT embeddings for an approximate model.
