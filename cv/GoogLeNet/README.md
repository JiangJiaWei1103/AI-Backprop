# Going Deeper with Convolutions (Szegedy et al., 2014)
Contributor: JiaWei Jiang
[[paper](https://wmathor.com/usr/uploads/2020/01/3184187721.pdf)]
## Challenges
* Immoderately increasing the model size (either depth or width) makes it prone to **overfitting**.
	* The labeled data is limited.
* Increased model complexity leads to higher computational cost.
	* The added complexity may be inefficient (*e.g.,* most weights are close to zero).
## Contributions
* Propose **Inception** module, which can capture **multi-scale** features by multi-branched convolutions with different kernel sizes.
* Apply **dimension reduction** through 1x1 convolutions.
	* Avoid exploding computational cost after feature maps are concatenated along channel dimension.
## Methodology
[![Screenshot-2024-04-27-at-10-30-53-PM.png](https://i.postimg.cc/y8Y9QZZk/Screenshot-2024-04-27-at-10-30-53-PM.png)](https://postimg.cc/PP9PNNKj)
![[Pasted image 20240427223357.png]]
* Apply ReLU after all convolutions.
	* Introduce additional non-linearity.
* Use [[average_pooling|average pooling]] in the output module (following [Lin et al.](https://arxiv.org/abs/1312.4400)).
	* Improve top-1 accuracy by about 0.6%.
* Add two auxiliary classifiers.
	* Enable gradients to effectively propagate back to lower layers.
		* Combat the [[vanishing_gradient|vanishing gradient]] problem.
## Discussion 
### How can 1x1 convolutions help reduce model size in term of number of parameters?
[![inception-3x3-drawio.png](https://i.postimg.cc/vBD50XQZ/inception-3x3-drawio.png)](https://postimg.cc/JDVynNDf)
[![inception-1x1-3x3-drawio.png](https://i.postimg.cc/x8MJZ4Bn/inception-1x1-3x3-drawio.png)](https://postimg.cc/xNj1Jt04)
As illustrated above, introducing 1x1 convolutions for dimension reduction can help reduce the number of parameters.
* 3x3 convolutions only: $[(3 \times 3) \times 192 + 1] \times 128 = 221312$ 
* 1x1 before 3x3: $[(1 \times 1) \times 192 + 1] \times 64 + [(3 \times 3) \times 64 + 1] \times 128 = 86208$
### Why can auxiliary classifiers provide additional regularization?

## Terminologies
* Hebbian principle
* Sparse structure approximation ([Arora et al.](https://arxiv.org/pdf/1310.6343))