# CBAM: Convolutional Block Attention Module (Woo et al., 2018)
Contributor: JiaWei Jiang <br>
[[paper](https://arxiv.org/pdf/1807.06521)] [[code](https://github.com/Jongchan/attention-module)]

## Background
Researches on convolutional neural networks (CNNs) mainly investigate three crucial factors of models, including **depth** (*e.g.,* VGGNet, ResNet), width (*e.g.,* [[GoogLeNet]]), and cardinality (the number of parallel branches) (*e.g.,* [[Xception]], ResNeXt).
## Challenges
* Focusing on depth, width, and cardinality inevitably increases the model capacity and computational overhead.
	* Note that **cardinality** is a more parameter-efficient option to improve the representation power of the model.
* Existing attention-based CNNs are computation-intensive (*e.g.,* deriving a 3D attention map) or don't fully exploit spatial and channel information.
## Contributions
* Propose a lightweight **C**onvolutional **B**lock **A**ttention **M**odule **CBAM**, which improves the representation power of CNNs in a plug-and-play manner.
	* Decompose the attention process to exploit channel and spatial information **sequentially**.
	* Can be incorporated into any CNN architectures with negligible overhead in terms of both parameter counts and [[floating_point_operations|GFLOPs]].
## Methodology
[![Screenshot-2024-05-31-at-4-14-24-PM.png](https://i.postimg.cc/Y0r8Rr9x/Screenshot-2024-05-31-at-4-14-24-PM.png)](https://postimg.cc/hzY9PqVJ)

Let  $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$ be an intermediate feature map, CBAM can be defined as follows,
```math
	\begin{align}
	\mathbf{F'} &= \mathbf{M_c}(\mathbf{F}) \odot \mathbf{F} \\
	\mathbf{F''} &= \mathbf{M_s}(\mathbf{F'}) \odot \mathbf{F'} \\
	\end{align}
```
where $\mathbf{M_c} \in \mathbb{R}^{C \times 1 \times 1}$ and $\mathbf{M_s} \in \mathbb{R}^{1 \times H \times W}$ are channel and spatial attention maps, respectively.

[![Screenshot-2024-05-31-at-4-14-41-PM.png](https://i.postimg.cc/GmF8LkzH/Screenshot-2024-05-31-at-4-14-41-PM.png)](https://postimg.cc/hXGPMQkB)
### Channel Attention Module
##### - Learn "what" to attend.
The channel attention module generates $\mathbf{M_c} \in \mathbb{R}^{C \times 1 \times 1}$ by exploiting the **inter-channel** relationship of features, which is shown below,

[![cbam-ch-attn.png](https://i.postimg.cc/yNLvjfWZ/cbam-ch-attn.png)](https://postimg.cc/Nyr6fk6s)

The channel attention can be defined as follows,
```math
	\mathbf{M_c}(\mathbf{F}) = \sigma(\mathbf{W_1}(\mathbf{W_0}(\mathbf{F^c_{avg}})) + \mathbf{W_1}(\mathbf{W_0}(\mathbf{F^c_{max}})))
```
where $\mathbf{F^c_{avg}} \in \mathbb{R}^{C \times 1 \times 1}$ and $\mathbf{F^c_{max}} \in \mathbb{R}^{C \times 1 \times 1}$ are spatial context descriptors obtained by average- and **max-pooling** (underexplored in previous literatures) operations. In order to **reduce parameter count**, the MLP is shared and the bottleneck hidden layer is adopted. Also, the ReLU activation is applied after $\mathbf{W_0}$.
### Spatial Attention Module
##### - Learn "where" to attend.
The channel attention module produces $\mathbf{M_s} \in \mathbb{R}^{1 \times H \times W}$ by exploiting the **inter-spatial** relationship of features, which is shown below,

[![cbam-spatial-attn.png](https://i.postimg.cc/dtBkhD15/cbam-spatial-attn.png)](https://postimg.cc/CzRLthb8)

The spatial attention can be defined as follows,
```math
	\mathbf{M_s}(\mathbf{F}) = \sigma(f^{7 \times 7} ([\mathbf{F^c_{avg}}; \mathbf{F^c_{max}}])
```
where $\mathbf{F^c_{avg}} \in \mathbb{R}^{1 \times H \times W}$ and $\mathbf{F^c_{max}} \in \mathbb{R}^{1 \times H \times W}$ are channel context descriptors obtained by average- and max-pooling operations, which can be thought of as **non-trainable** 1x1 convolutions.<br>
To sum up, two attention modules are applied sequentially in a **channel-first** manner.

[![Screenshot-2024-05-31-at-4-14-54-PM.png](https://i.postimg.cc/G34jc5Dt/Screenshot-2024-05-31-at-4-14-54-PM.png)](https://postimg.cc/F1QSpGDX)
## Discussion 
### Why do authors apply the sigmoid function to attention maps, instead of the softmax?
## Terminologies
* Grad-CAM ([Selvaraju et al.](https://arxiv.org/pdf/1610.02391))