# PCA-Gauss

PCA-Gauss is a system for non-interactive differentially-private data release. The implementation is based on the following paper:

Chanyaswad, Thee, Changchang Liu, and Prateek Mittal. "Coupling Dimensionality Reduction with Generative Model for Non-Interactive Private Data Release." arXiv preprint arXiv:1709.00054 (2017).

Hence, please refer to this paper for more detail on the PCA-Gauss system.

The implementation is for using DPPCA with the Gaussian model for all three algorithms proposed in the paper.

## How to:

The main class to run the system is the ‘PCA-Gauss’ class. This class is initialized with three parameters:
algorithm = ‘supervised’, ‘unsupervised’, or ‘gmm’;
epilonGauss = the epsilon value used for fitting the Gaussian model.
epsilonPca = the epsilon value used for DPPCA.

The main method is the ‘.generate_dpdata’, which takes in, among others, the private data and the dimension to reduce the data to. It then returns the differentially-private synthesized data according to the parameters set in the initialization.

Some details on the inputs of ‘.generate_dpdata’:
1) For the ’supervised’ algorithm, yPrivate is required. For the ‘unsupervised’ algorithm, only xPrivate is needed. For the ‘gmm’ algorithm, yPrivate is required and should be the class label.
2) Recall from the paper that the input private data are presumed to have zero mean.
3) The parameter ‘a’ is the range of the the private data, as defined in the paper (it is probably the easiest if you ensure that your data lie in [-1,1], and, then, the default a=1.0 is fine).
4) The parameter ‘seed’ is the seed for the randomization generator.


### Prerequisites

The implementation is in Python 2 and numpy is required.


## Authors

* **Thee Chanyaswad**

