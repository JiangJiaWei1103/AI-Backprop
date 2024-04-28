# Xception: Deep Learning with Depthwise Separable Convolutions (Chollet, 2017)
Contributor: JiaWei Jiang <br>
[[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf)]
## Challenges
* Based on the original [[GoogLeNet|Inception]] hypothesis, the author tries to answer following questions,
	* What's the effect of the channel group size for each spatial convolution?
	* Can cross-channel and spatial correlations be mapped completely separately?
## Contributions
* Interpret Inception modules as an intermediate step between canonical convolutions and **depthwise separable convolutions** ([Mamalet and Garcia](https://liris.cnrs.fr/Documents/Liris-5659.pdf) or earlier).
* Propose **Xception** (Extreme Inception), entirely based on **depthwise separable convolution** layers.
	* Completely decouple the mapping of cross-channel and spatial correlations. 
	* Outperform Inception V3, which has a similar parameter count.
## Methodology
[![Screenshot-2024-04-28-at-8-06-47-PM.png](https://i.postimg.cc/W4wx4nwm/Screenshot-2024-04-28-at-8-06-47-PM.png)](https://postimg.cc/rdzfn5bz)
## Discussion
### The Discrete Spectrum of Convolutions
[![xception-spectrum-drawio.png](https://i.postimg.cc/Pxxdg4Lx/xception-spectrum-drawio.png)](https://postimg.cc/1gkx6Drh)
### The Difference btw Extreme Version of Inception and Depthwise Separable Convolution
|                   | Extreme Version of Inception | Depthwise Separable Convolution                             |
| ----------------- | ---------------------------- | ----------------------------------------------------------- |
| Ordering of convs | Pointwise then depthwise     | Depthwise then pointwise                                    |
| Non-linearity     | ReLU after both convs        | Usually without non-linearities btw depthwise and pointwise |
[![Screenshot-2024-04-28-at-8-53-40-PM.png](https://i.postimg.cc/WpyR5ZM2/Screenshot-2024-04-28-at-8-53-40-PM.png)](https://postimg.cc/RqwbFNs8)

> It may be that **the depth of the intermediate feature spaces** on which spatial convolutions are applied is critical to the usefulness of the non-linearity.

Note that Inception modules map cross-channel correlation first, which can create deep latent feature spaces.
## Terminologies
* Spatially separable convolution
* Transformation-invariant scattering
* 