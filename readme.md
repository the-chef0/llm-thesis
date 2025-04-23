# Earlier Notes on Pruning and Performance Metrics
[Click here to navigate to the earlier notes.](./earlier-nodes.md)

# Brief Project Proposal
[Click here for the full proposal document.](./proposal.pdf)

Pruning an LLM is can be a straightforward task in some cases, but not in others. Simple LLMs like the first Llama models only have layers of chained transformer blocks with the same input and output dimension everywhere, making it easy to remove a single layer [(paper 3)](#literature). Llama-2 has residual/skip connections, so removing a layer is more challenging but still doable - the residual connection needs to be "rerouted" to the following layer. 

In other and newer models, it becomes considerably more challenging. Multimodal LLMs might contain convolutional layers, where removing one could cause a serious dimension mismatch [(link)](https://github.com/liyunqianggyn/Awesome-LLMs-Pruning/blob/main/concepts/other_concepts.md). Mixture-of-Experts models present yet another set of challenges, and one solution is to prune based on hints from the router [(paper 11)](#literature).

It seems like there is a gap in the field - there does not yet exist a general LLM pruning methodology that could be applicable on a wide range of architectures. The purpose of the thesis is to try to come up with such a methodology, and use it to conduct some broader LLM compression experiments.

Pruning is one way to compress an LLM and quantization is another. They seem not to be mutually exclusive, in the sense that the amount of pruning one does impacts the amount of quantization one can do, and vice versa. This indicates that when compressing an LLM, there could be an optimal mixture of both that minimizes size while maximizing performance.

There already exist open-source tools that collect much of the wide range of quantization techniques into one codebase, e.g. [LLaMa-Factory](#llama-factory). We can use such tools in combination with an implementation of our novel LLM pruning methodology to conduct experiments and learn more about this "pruning vs. quantization" tradeoff.

# Project plan

|   Deadline   |                                                                                                           Milestones                                                                                                           |
|:------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 22 April     | Thesis start                                                                                                                                                                                                                   |
| 31 May       | Completed review of LLM architectures and related pruning strategies <br> Completed classification/taxonomy of of LLM pruning strategies <br> Completed relevant thesis sections                                               |
| 30 June      | Completed implementation of LLM pruning toolkit <br> Completed experimental design for first iteration of experiments <br> Completed relevant thesis sections                                                                  |
| 8 July       | First Stage Evaluation                                                                                                                                                                                                         |
| 31 August    | Completed first iteration of experiments and obtained results <br> Processed experiment results <br> Completed relevant thesis sections <br> Completed experimental design for second iteration of experiments (if applicable) |
| 30 September | Completed second iteration of experiments and obtained results <br> Processed experiment results <br> Completed relevant thesis sections and first thesis draft                                                                |
| 7 October    | Green Light Review                                                                                                                                                                                                             |
| 17 November  | Completed final thesis draft                                                                                                                                                                                                   |
| 18 November  | Thesis Defense Evaluation                                                                                                                                                                                                      |

# Tools
## LLaMa-Factory
https://github.com/hiyouga/LLaMA-Factory
An LLM compression toolkit.

## Towards Any Structural Pruning
https://github.com/VainF/Torch-Pruning

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

Upon computing the NRI of each layer and removing a fixed fraction of the least important ones, there was a considerable amount of Attention and FFN layers removed that were not together in one transformer block. It seems like this sublayer perspective might also be useful

10. [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710)

This is a more dynamic take on the idea of pruning or skipping later layers. Instead of statically pruning some last $k$ layers, the number of layers to skip $k$ is determined at inference time as a function of the input. They train a model with layer dropout, assign higher dropout rates to later layers, and train with a loss function that penalizes late exits.

11. [MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router](https://arxiv.org/abs/2410.12013)

This is another dynamic, input dependent technique that prunes expert weights rather than Attention weights. The pruning metric is designed such that expert weights that get utilized the least are more likely to get pruned. The magnitude of any given weight is multiplied by the output of the gate times the element of the token that the weight affects.

12. [A Survey on Model Compression for Large Language Models](https://arxiv.org/abs/2308.07633)

Contains a very good taxonomy of existing compression techniques and some experimental results that show their effectiveness.

13. [Efficient Large Language Models: A Survey](https://arxiv.org/abs/2312.03863)

Contains a superset of the taxonomy in paper 12, offering a wider and more fine-grained breakdown, and including efficiency-oriented frameworks.

14. [A Survey on Multimodal Large Language Models](https://arxiv.org/abs/2306.13549)

A very nice survey with lots of nice figures that describes how MLLMs are structured, and what networks are used for which components.

15. [BitNet b1.58 2B4T Technical Report](https://arxiv.org/abs/2504.12285)

Microsoft released a tiny 1GB 1.58-bit model that can be run on CPU and is claimed to compete with some other small models with similar parameter counts but higher resource usage.