# Squeezeformer: An Efficient Transformer for Automatic Speech Recognition (Kim et al., 2022)
Contributor: JiaWei Jiang <br>
[[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/3ccf6da39eeb8fefc8bbb1b0124adbd1-Paper-Conference.pdf)]
<br>
Conformer architecture ([Gultai et al.](https://arxiv.org/pdf/2005.08100)) has become the *de facto* backbone for several downstream speech tasks, which can simultaneously capture local and global context from audio signals by its proposed **convolution-augmented** Transformer architecture.
## Challenges
* MHA mechanism suffers from **quadratic** complexity, limiting model efficiency especially on long sequence.
* Conformer architecture is more complicated than the canonical Transformers.
	* *e.g.,* Different normalization methods, multiple activation functions, Macaron structure, as well as back-to-back MHA and convolution modules.
	* The design may complicate hardware deployment (*e.g.,* edge devices).
## Contributions
* Re-examine macro- and micro-architecture of Conformer.
	* Question the necessity of its design choices.
* Propose **Squeezeformer**, having a more efficient **hybrid attention-convolution** architecture.
* Outperform SOTA ASR models (evaluated by word-error-rate (WER)) with promising scalability in terms of parameter count and FLOPs.
## Methodology
[![squeezeformer-fig2.png](https://i.postimg.cc/KvjsR0P6/squeezeformer-fig2.png)<br>
[![squeezeformer-fig4.png](https://i.postimg.cc/yxfgQwPB/squeezeformer-fig4.png)](https://postimg.cc/zHgG326P)<br>
* (Macro) Incorporate **U-Net** structure ([Ronneberger et al.](https://arxiv.org/pdf/1505.04597), [Perslev et al.](https://proceedings.neurips.cc/paper/2019/file/57bafb2c2dfeefba931bb03a835b1fa9-Paper.pdf)).
	* Reduce **temporal redundancy** (in terms of cosine similarity of embeddings of neighboring speech frames) and save compute.
	* **Upsampling** is essential to training stability.
* (Macro) Adopt the design paradigm of canonical Transformer blocks.
	* Consider the convolution module as a **local MHA**.
		* From back-to-back FMCF (C with large kernel size, mixed up with M) to MF/CF.
	* Drop the Macaron structure.
		* Back to MHA then FF.
* (Micro) Unify activation functions to **Swish** only.
	* Make it simpler for hardware deployment.
* (Micro) Simplify LayerNorm by adopting **scaled postLN** (postLN then scaling).
	* Remove redundant LN (back-to-back postLN and preLN).
	* Unify LN schemes.
	* Stabilize training by incorporating a scaling layer.
		*  Because authors observe preLN scales down the input signal by analyzing norms of learnable scale variables.
* (Micro) Use [[Xception|depthwise separable convolution]] for the 2nd convolution operation in the sub-sampling block.
	* Reduce parameter count and FLOPs.
## Discussion
## Terminologies
* Macaron structure ([Lu et al.](https://arxiv.org/pdf/1906.02762))
* RNN-Transducer (RNN-T) ([Graves](https://arxiv.org/pdf/1211.3711))
* Connectionist Temporal Classification (CTC) ([Graves et al.](https://www.cs.toronto.edu/~graves/icml_2006.pdf))
