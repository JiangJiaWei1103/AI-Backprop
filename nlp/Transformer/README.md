# Attention is All You Need (Vaswani et al., 2017)
Contributor: JiaWei Jiang <br>
[[paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)]
## Background
## Challenges
## Contributions
## Methodology
### Positional Encoding
Positional encoding is used to inject the concept of **order** into a sequence, enriching a sequence with **relative or absolute position** information of each token (*e.g.,* word tokens, phonetic tokens).
* Preserve the semantic of a sequence.
```
1. I like this movie because I don't think it's too mind-twisting.
2. I don't like this movie because I think it's too mind-twisting.
```
As can be observed in the example above, the semantic of two sentences are opposite. If **order** information is ditched, attention mechanism will fail to distinguish between the two.<br>
Instead of using a single number per token, a position is represented by a $d$-dimensional **vector**. To better understand the meaning of PE, the formula is rewritten as follows,
$$
	\begin{align}
	PE_{(t, 2i)} &= sin(\omega_{i}t) \\
	PE_{(t, 2i+1)} &= cos(\omega_{i}t) \\
	\omega_i &= \frac{1}{10000^{2i / d}}
	\end{align}
$$
, where $t$ is the time index along a sequence and $i \in [0, \frac{d}{2})$ denotes the vector dimension index. As $i$ gets up along the vector dimension, the angular frequency goes down (*i.e.,* wavelength $\lambda$ gets longer). Note that the possibility of **encoding collision** is higher for a greater frequency $\omega_i$, because oscillation (bit altering) is more sensitive to a small change of $t$.<br>
Now, let $d = 512$, the positional encoding $e_t \in \mathbb{R}^{d}$ at $t$ can be represented as follows,
$$
	e_t = 
	\begin{bmatrix}
	sin(\frac{t}{10000^{2 \times 0 / 512}}) \\ 
	cos(\frac{t}{10000^{2 \times 0 / 512}}) \\ 
	sin(\frac{t}{10000^{2 \times 1 / 512}}) \\ 
	cos(\frac{t}{10000^{2 \times 1 / 512}}) \\
	. \\ . \\ . \\
	sin(\frac{t}{10000^{2 \times 255 / 512}}) \\ 
	cos(\frac{t}{10000^{2 \times 255 / 512}}) \\ 
	\end{bmatrix}
$$
## Discussion
### Why do Authors Use 10000 in PE?
[![pe-e128-t64-base10000.png](https://i.postimg.cc/0Qwp0Jjb/pe-e128-t64-base10000.png)](https://postimg.cc/ZBJBTnrS)
<br>
[![pe-e128-t64-base100.png](https://i.postimg.cc/prQzgbDv/pe-e128-t64-base100.png)](https://postimg.cc/PvxCw73V)
<br>
[![pe-e128-t64-base1.png](https://i.postimg.cc/W4DrsJMX/pe-e128-t64-base1.png)](https://postimg.cc/67XT0qbv)
<br>
Above illustrates the effect of selecting different base (denominator, from $10000$ to $1$) to derive the angular frequency in PE formula. Observations are summarized as follows,
1. Choosing a large denominator (10000 in the paper) somewhat guarantees that each position can have an **unique** positional encoding without colliding with the others.
2. Cosine similarities among position pairs **decrease** as distances get **farther**.
3. **Symmetric** property is shown in cosine similarity matrices.
### Associate Positional Encoding with Binary Encoding
## Terminologies
## References
* [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/#the-intuition)
