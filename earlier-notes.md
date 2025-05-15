# LLM evaluation metrics

## Perplexity

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

## CommonSenseQA
https://paperswithcode.com/paper/commonsenseqa-a-question-answering-challenge

A multiple-choice, common sense benchmark, measured as the percentage of correct answers.

## Measuring Massive Multitask Language Understanding (MMLU)
https://huggingface.co/datasets/cais/mmlu

Another multiple choice benchmark, including questions from a large range of topics, including but not limited to STEM subjects, humanities, medicine and law. Measured as the percentage of correct answer.

# Quantization techniques and tools

Weights are quantized using symmetric or asymmetric quantization. Activations are quantized with either **dynamic** or **static** quantization. Dynamic quantization calculates statistics of activations after each layer, and uses them to find a zeropoint and scale factor to quantize. Static quantization uses a calibration dataset to collect distribution statistics of activations, computes the quantization parameters based on those, and uses those same ones every time during inference.

## GPTQ
Let $W$ be a weight matrix and let $H = \frac{\partial^2 \mathcal{L}}{\partial W^2}$ be its Hessian matrix. Let $w_{ij}$ denote an FP32 weight, let $w^q_{ij}$ denote its INT4 quantization, and let $w^{-q}_{ij}$ denote its dequantization from INT4 back to FP32. Then for all rows $i$
 - $\Delta q = \frac{w_{i1} - w^{-q}_{i1}}{h_{i1}}$
 - $w^{q}_{ij} = w_{ij} + \Delta q h_{ij}$ for all columns $j$

 That is, weights in a given row are updated based on how sensitive to quantization the value in the first column is.

## GGUF
Allows to offload layers onto the CPU. It does this by splitting a layer into superblocks, each containing a set of subblocks. Scale factors are computed for the superblock, and for each subblock. The subblock scale factors and quantized using the superblock scale factor, and the quantized subblock scale factors are used to quantize the weights.

https://huggingface.co/docs/hub/en/gguf#quantization-types

There is a wide variety of quantized models on HuggingFace, using a variety of GGUF methods. Some of these are marked as legacy, i.e. "not used widely as of today". Here I summarize the non-legacy ones.

A model whose quantization is denoted as $QbK$ (with $b$ the number of bits) represents $b$-bit quantization that uses blocks of multiple weights that each share a scale factor and a bias. 

This seems vaguely similar to the ideas in [(2)](#literature), except the MX formats have no bias for a whole block, and some of the bit widths in $QbK$ do not seem MX-compliant. $QbK$ also allows for things like $b=3$, even though 3 bit quantization is not specified in MX specification.

A model whose quantization is denoted as $IQbs$ ($s \in \{XXS, XS, S, M, NL\}$) represents $b$-bit quantization where the weights are also scaled by an importance matrix. It seems like $s$ somehow denotes the precision of the importance matrix, but I was not yet able to find out how exactly. I presume the importance matrix is something like an inverse Hessian matrix described in [(4)](#literature).

The above quantization types conform to a format called GGUF. It seems that the purpose of the MX format is to physically store the bits in a "block with shared scale" format, whereas the GGUF format aims to quantize models by splitting existing weights into blocks and factoring out a scale.