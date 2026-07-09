# Transformer Paper (Attention Is All You Need) Test Queries

**Document Details:** [Insert Transformer PDF Path Here]
**Target Chunk Size:** [TBD]
**Tokens/Words per chunk:** [TBD]

## 1. Semantic Queries (Conceptual, Abstract)
1. What is the fundamental limitation of recurrent neural networks that the Transformer architecture attempts to solve?
2. How does the concept of "self-attention" allow the model to understand context better than convolutional layers?
3. What is the theoretical purpose of using multi-head attention rather than a single attention mechanism?
4. How does the model prevent leftward information flow in the decoder to preserve the auto-regressive property?
5. What role does positional encoding play in a model that lacks recurrence and convolution?
6. How does the architecture balance the trade-off between computational complexity and maximum path length?
7. In what ways does the Transformer architecture improve training parallelization?
8. Why is scaled dot-product attention preferred over additive attention in this architecture?
9. How does label smoothing affect the model's perplexity and accuracy during training?
10. What is the significance of the residual connections around each of the sub-layers?

## 2. Keyword Queries (Specific Facts, Math, Parameters)
11. What is the exact mathematical formula for Scaled Dot-Product Attention?
12. What is the dimension size ($d_{model}$) used in the base model?
13. How many attention heads ($h$) are used in the standard Transformer architecture?
14. What are the specific frequencies used in the sine and cosine functions for positional encoding?
15. What specific optimizer is used, and what are its $\beta_1$ and $\beta_2$ parameters?
16. What was the exact BLEU score achieved by the Transformer (big) on the WMT 2014 English-to-French translation task?
17. What is the dropout rate applied to the output of each sub-layer before it is added to the sub-layer input?
18. How many identical layers ($N$) are in the encoder and decoder stacks of the base model?
19. What specific hardware (GPU type and count) was used to train the base models?
20. What is the formula used to vary the learning rate during training?

## 3. Hybrid Queries (Context + Specific Details)
21. When comparing self-attention to recurrent layers, how does the maximum path length $O(1)$ specifically impact the learning of long-range dependencies?
22. How does the scaling factor of $\frac{1}{\sqrt{d_k}}$ specifically prevent the softmax function from entering regions with extremely small gradients?
23. In the context of multi-head attention, how are the specific linear projections of queries, keys, and values combined?
24. How does the specific byte-pair encoding (BPE) vocabulary size affect the model's performance on the WMT 2014 English-to-German task?
25. When utilizing the Adam optimizer, how does the warmup_steps parameter specifically alter the learning rate curve during the first few epochs?
26. How do the specific dimensions of the feed-forward networks ($d_{ff} = 2048$) relate to the $d_{model}$ dimensionality?
27. In what specific way does the masking in the decoder self-attention layer differ from the encoder self-attention layer?
28. How does the specific choice of sinusoidal positional encodings allow the model to extrapolate to sequence lengths longer than those encountered during training?
29. When analyzing computational complexity, how does the $O(n^2 \cdot d)$ complexity of self-attention compare to recurrent layers when sequence length $n$ is smaller than representation dimensionality $d$?
30. How did the application of label smoothing with $\epsilon_{ls} = 0.1$ specifically impact the model's confidence in its predictions?
