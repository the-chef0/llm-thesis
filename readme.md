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

A model whose quantization is denoted as $Qb`_`K$ (with $b$ the number of bits) represents $b$-bit quantization that uses blocks of multiple weights that each share a scale factor and a bias. 

This seems vaguely similar to the ideas in [(2)](#literature), except the MX formats have no bias for a whole block, and some of the bit widths in $Qb\_K$ do not seem MX-compliant. $Qb`_`K$ also allows for things like $b=3$, even though 3 bit quantization is not specified in MX specification.

A model whose quantization is denoted as $IQb`_`s$ ($s \in \{XXS, XS, S, M, NL\}$) represents $b$-bit quantization where the weights are also scaled by an importance matrix. It seems like $s$ somehow denotes the precision of the importance matrix, but I was not yet able to find out how exactly. I presume the importance matrix is something like an inverse Hessian matrix described in [(4)](#literature).

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