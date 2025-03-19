# Perplexity

https://huggingface.co/docs/transformers/en/perplexity
https://thegradient.pub/understanding-evaluation-metrics-for-language-models/

Let $X = (x_0, x_1, ..., x_t)$ be a tokenized sequence. The perplexity of $X$ is 
$\text{PPL}(X) = \exp (-\frac{1}{t} \sum_{i}^{t} \log p_\theta (x_i | x_{<i})) = \exp{H(X)}$.

Normally, cross-entropy between two distributions is $H(p,q) = - \sum_x p(x) \log q(x)$, but in LLMs we use the tokens that have already been predicted vs. the predicted distribution for the next token, leading to the $H(X)$ above.

For a given model, perplexity is maximized when entropy is maximized, and entropy is maximized when the model can't do better than predict a uniform distribution over all possible tokens in the vocabulary $V$ with size $|V|$.

It follows from Shannon entropy that
$H(X) \leq -|V| \times \frac{1}{|V|} \times \log{(\frac{1}{|V|})} = \log{(|V|)}$. That is, if a model can't do better than a uniform distribution for the next token, entropy is bounded from above by $\log{(|V|)}$.

This also means that $p_\theta(x_i | x_{<i}) \geq \frac{1}{|V|}$, which is dependent on the size of the model's vocabulary. This has two consequences:

1. It does not make sense to compare perplexities of two models if their vocabularies have different sizes, unless there is a way to transform them. 
2. Maybe we can evaluate how "good" or "bad" a change in perplexity is by comparing it to the upper bound, as determined by the model's vocabulary size.

# Types of quantized models

https://huggingface.co/docs/hub/en/gguf#quantization-types

There is a wide variety of quantized models on HuggingFace, using a variety of quantization techniques. Some of these are marked as legacy, i.e. "not used widely as of today". Here I summarize the non-legacy ones.

A model whose quantization is denoted as $QbK$ (with $b$ the number of bits) represents $b$-bit quantization that uses blocks of multiple weights that each share a scale factor and a bias. 

This seems vaguely similar to the ideas in [(2)](#literature), except the MX formats have no bias for a whole block, and some of the bit widths in $QbK$ do not seem MX-compliant. $QbK$ also allows for things like $b=3$, even though 3 bit quantization is not specified in MX specification.

A model whose quantization is denoted as $IQbs$ ($s \in \{XXS, XS, S, M, NL\}$) represents $b$-bit quantization where the weights are also scaled by an importance matrix. It seems like $s$ somehow denotes the precision of the importance matrix, but I was not yet able to find out how exactly. I presume the importance matrix is something like an inverse Hessian matrix described in [(4)](#literature).

The above quantization types conform to a format called GGUF. It seems that the purpose of the MX format is to physically store the bits in a "block with shared scale" format, whereas the GGUF format aims to quantize models by splitting existing weights into blocks and factoring out a scale.

# Literature

1. [FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design](http://arxiv.org/abs/2401.14112)

Talks about how FP6 quantization seems to be a good middle ground between FP4 and FP8 in terms of performance (measured as perplexity, see [above](#perplexity)). The problem is that 6 is an irregular bit width, which makes hardware implementation challenging. The paper mainly focuses on the hardware (e.g. GPU, TPU) design challenges associated with FP6 quantization, and proposes solutions.

2. [OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

Defines the "MX-complaint" format and low-precision datatypes conforming to it - MXINT8, MXFP4, MXFP6 and MXFP8. The format consists of $k$ private elements $P_i$ 
($1 \leq i \leq k$), each $d$ bits long. All $k$ elements share one $w$-bit scale factor $X$. For example, FP4 comes in groups of $k=32$ elements, each $d=4$ bits long, with a shared $w=8$ bit scaling factor.

3. [ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](http://arxiv.org/abs/2403.03853)

Studies an observation that the Llama7B and Llama13B models contain layers (Attention + FFN) that make almost no change to the hidden representation. They define a metric (BI) for evaluating how "important" a layer is - it measures the cosine similarity between outputs of two consecutive layers. Turns out they were able to up to 20% of the "least important" layers before observing a sharp drop on the MMLU benchmark score. Perplexity seemed to increase roughly linearly.

4. [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)

5. [Pruning vs Quantization: Which is Better?](https://arxiv.org/pdf/2307.02973)

Tries to make general claims about quantization outperforming pruning in most neural network use cases. They obtain the results in an interesting way. First, they formulate the expected error from both quantization and pruning, Then they formulated general optimization problems representing the process of quantization and pruning. They use the expected error models to derive bounds on quantization/pruning error, and they use solutions of the optimization problems to validate the bounds empirically.

The result seems pretty compelling, but I wonder how well their mathematical model of post-training pruning applies to [paper 3](#literature). In this paper, they formulate post-training pruning as finding a weight matrix $w$ and a binary mask matrix $m$ such that $||X(m \cdot w) - Xw_\text{orig}||_2^2$ is minimized. In other words, "find new weights, along with a mask for setting some weights to 0, such that the outputs from the layer are as close as possible to what they are with the original weights". 

In [paper 3](#literature) however, they do not explicitly pick any such mask $m$ or optimize new weights $w$ to prune with $m$. They prune by simply ripping a layer out. If we think of "ripping the layer out" as replacing an attention layer with a unit matrix of compatible dimensions, it is possible in principle to represent this using the language used in this paper. We would have to pick optimal masks and weights to replace each matrix making up the attention layer ($Q$, $K$ and $V$), such that passing through the attention layer is equal to multiplying by an identity matrix (will write out more precisely later).

I'd be curious to see whether the pruning error from [paper 3](#literature) falls into the bounds that this paper establishes and is empirically consistent with it. This paper makes some strong claims, but it's a notably different perspective on pruning.

6. [SplitQuant: Layer Splitting for Low-Bit Neural Network
Quantization](https://arxiv.org/pdf/2501.12428)

Contains an interesting idea for improving low-precision quantization. Low-precision quantization can fail to capture outliers in weights due to its low resolution and setting the scale away from the outlier. However, removing outliers can reduce performance because they can activate for unusual input cases, improving decision boundaries. In this paper, they run k-means clustering on the weights to capture multiple "centers of mass", identifying the outliers as separate clusters. Then, they split the layer into mutiple low-precision quantized layers, quantizing each one with different ranges, such that the outlier cluster is covered too.

7. [The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/pdf/2403.17887)

Very similar results to [paper 3](#literature) but with one caveat: On knowledge retrieval benchmarks, performance does not start to drop until a non-trivial fraction of the deepest layers is pruned. On reasoning benchmarks however, performance starts to drop right away. This begs the question of if, how and where is the encoded information in LLM weights localized.

Also contains a technique for pruning layers with simple residual connections. Earlier, Yunqiang showed me some examples of models where it's not immediately obvious how to prune them because of complex residual connections. Maybe it's just a question of rerouting the residuals in some "correct" way.

8. [What Matters in Transformers? Not All Attention is Needed](https://openreview.net/pdf?id=YLTWwEjkdx)

Pretty much the same as [paper 3](#literature)?

9. [Rethinking the Impact of Heterogenous Sublayers in Transformers](https://openreview.net/pdf?id=qG1S5eXMzx)

A more fine-grained perspective on pruning compared to papers 3, 8 and 9. Both attention layers and FFN layers are considered independent candidates for pruning. For the metric, they take the relative difference in perplexity between the baseline model and the model with the given layer removed, and then they divide by the number of parameters in the layer - normalized relative impact (NRI). This gives more of a "impact per parameter" view.

Upon computing the NRI of each layer and removing a fixed fraction of the least important ones, there was a considerable amount of Attention and FFN layers removed that were not together in one transformer block. It seems like this sublayer perspective might also be useful.