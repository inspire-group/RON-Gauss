# RON-Gauss

RON-Gauss is a system for non-interactive differentially-private data release. The implementation is based on the following paper:

*Thee Chanyaswad, Changchang Liu, and Prateek Mittal. “RON-Gauss: Enhancing Utility in Non-Interactive Private Data Release,” Proceedings on Privacy Enhancing Technologies (PETS), vol. 2019, no. 1, 2018.* (https://content.sciendo.com/view/journals/popets/2019/1/article-p26.xml)

Hence, please refer to this paper for more detail on the RON-Gauss model.

The implementation is for all three algorithms proposed in the paper.

## How to:

The main class to run the system is the `RONGauss` class. This class is initialized with three parameters:
- `algorithm` = `unsupervised`, `supervised`, or `gmm`;
- `epilon_mean` = the epsilon value used for deriving the sample mean;
- `epsilon_cov` = the epsilon value used for deriving the sample covariance.

The main method is the `generate_dpdata`, which takes in, among others, the private data and the dimension to reduce the data to. It then returns the differentially-private synthesized data according to the parameters set in the initialization. There are two additional parameters for `generate_dpdata`:
- `reconstruct` = specify whether to reconstruct the synthetic data back to the original feature space after the projection;
- `centering` = specify whether to automatically center the synthesized data for the `unsupervised` and `supervised` algorithms. If `False` the synthesized data will have the mean equal to the differentiallly-private mean of the private data.

Some details on the inputs of `generate_dpdata`:
1) For the `unsupervised` algorithm, only the feature data matrix `X` is needed. 
2) For the `supervised` algorithm, in addition to `X`, the training label `y` is also required, along with its absolute range `max_y`. 
3) For the `gmm` algorithm, both `X` and `y` are required, and `y` should be the class label.


### Prerequisites

The implementation is in Python 2 and has been tested for compatibility with Python 3. The required packages are `numpy`, `scipy`, and `sklearn.preprocessing`.


## Author

* **Thee Chanyaswad**

