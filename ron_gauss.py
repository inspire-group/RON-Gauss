#
# ron_gauss.py
# Author: Thee Chanyaswad
#
# Version 1.0
# 	- Initial implementation.
#
# Version 1.1
# 	- Bug fixes.
# 	- Add reconstruction option.
# 	- Minor stylistic update.
#
# Version 1.2
# 	- Add mean adjustment option for unsupervised and supervised algs.
#
# Version 1.3 (submitted by mlopatka; 17-07-2019)
#  	- Enforce python black formatting guidelines.
#   -

import numpy as np
import scipy
from sklearn import preprocessing


class RONGauss:
    """TO-DO: Add class description
    """
    def __init__(self, algorithm="supervised", epsilon_mean=1.0, epsilon_cov=1.0):
        self.algorithm = algorithm
        self.epsilon_mean = epsilon_mean
        self.epsilon_cov = epsilon_cov

    def generate_dpdata(
        self,
        X,
        dimension,
        y=None,
        n_samples=None,
        max_y=None,
        reconstruct=False,
        centering=False,
        prng_seed=None,
    ):
        
        if self.algorithm == "unsupervised":
            x_dp = self._unsupervised_rongauss(X, dimension, n_samples, reconstruct, centering, prng_seed)
            y_dp = None

        elif self.algorithm == "supervised":
            x_dp, y_dp = self._supervised_rongauss(X, dimension, y, n_samples, max_y, reconstruct, centering, prng_seed)

        elif self.algorithm == "gmm":
            x_dp, y_dp = self._gmm_rongauss(X,dimension, y, n_samples, reconstruct, prng_seed)
        
        return (x_dp, y_dp)
    
    def _unsupervised_rongauss(
        self,
        X,
        dimension,
        n_samples,
        reconstruct,
        centering,
        prng_seed,
    ):
        prng = np.random.RandomState(prng_seed)
        (x_bar, mu_dp) = self._data_preprocessing(X, self.epsilon_mean)
        (x_tilda, proj_matrix) = self._ron_projection(x_bar, dimension)

        (N, P) = x_tilda.shape
        noise_var = (2.0 * np.sqrt(P)) / (N * self.epsilon_cov)
        if n_samples is None:
            num_samples = N
        else:
            num_samples = n_samples
        cov_matrix = np.inner(x_tilda.T, x_tilda.T) / N
        laplace_noise = prng.laplace(scale=noise_var, size=(P, P))
        cov_dp = cov_matrix + laplace_noise
        synth_data = prng.multivariate_normal(np.zeros(P), cov_dp, num_samples)
        x_dp = synth_data
        if reconstruct:
            x_dp = self._reconstruction(x_dp, proj_matrix)
        else:
            #project the mean down to the lower dimention
            mu_dp = np.inner(mu_dp, proj_matrix)
        self._mu_dp = mu_dp

        if not centering:
            x_dp = x_dp + mu_dp
        return x_dp

    def _supervised_rongauss(
        self,
        X,
        dimension,
        y,
        n_samples,
        max_y,
        reconstruct,
        centering,
        prng_seed,
    ):  
        prng = np.random.RandomState(prng_seed)
        (x_bar, mu_dp) = self._data_preprocessing(X, self.epsilon_mean)
        (x_tilda, proj_matrix) = self._ron_projection(x_bar, dimension)

        (N, P) = x_tilda.shape
        noise_var = (2.0 * np.sqrt(P) + 4.0 * np.sqrt(P) * max_y + max_y ** 2) / (
            N * self.epsilon_cov
        )
        if n_samples is None:
            num_samples = N
        else:
            num_samples = n_samples
        y_reshaped = y.reshape(len(y), 1)
        augmented_mat = np.hstack((x_tilda, y_reshaped))
        cov_matrix = np.inner(augmented_mat.T, augmented_mat.T) / N
        laplace_noise = prng.laplace(scale=noise_var, size=cov_matrix.shape)
        cov_dp = cov_matrix + laplace_noise

        synth_data = prng.multivariate_normal(np.zeros(P + 1), cov_dp, num_samples)
        x_dp = synth_data[:, 0:-1]
        y_dp = synth_data[:, -1]
        if reconstruct:
            x_dp = self._reconstruction(x_dp, proj_matrix)
        else:
            #project the mean down to the lower dimention
            mu_dp = np.inner(mu_dp, proj_matrix)
        self._mu_dp = mu_dp

        if not centering:
            x_dp = x_dp + mu_dp
        
        return (x_dp, y_dp)

    def _gmm_rongauss(
        self,
        X,
        dimension,
        y,
        n_samples,
        reconstruct,
        prng_seed,
    ):
        prng = np.random.RandomState(prng_seed)
        syn_x = np.array([])
        syn_y = np.array([])
        for label in np.unique(y):
            idx = np.where(y == label)
            x_class = X[idx]
            (x_bar, mu_dp) = self._data_preprocessing(x_class, self.epsilon_mean)
            (x_tilda, proj_matrix) = self._ron_projection(x_bar, dimension)

            (N, P) = x_tilda.shape
            noise_var = (2.0 * np.sqrt(P)) / (N * self.epsilon_cov)
            if n_samples is None:
                num_samples = N
            else:
                num_samples = n_samples
            mu_dp_tilda = np.inner(mu_dp, proj_matrix)
            cov_matrix = np.inner(x_tilda.T, x_tilda.T) / N
            laplace_noise = prng.laplace(scale=noise_var, size=(P, P))
            cov_dp = cov_matrix + laplace_noise
            synth_data = prng.multivariate_normal(mu_dp_tilda, cov_dp, num_samples)
            if reconstruct:
                synth_data = self._reconstruction(synth_data, proj_matrix)
            if len(syn_x) == 0:
                syn_x = synth_data
            else:
                syn_x = np.vstack((syn_x, synth_data))

            syn_y = np.append(syn_y, label * np.ones(num_samples))
        
        return syn_x, syn_y

    @staticmethod
    def _data_preprocessing(X, epsMu, prng_seed=None):
        (N, M) = X.shape
        # pre-normalize
        Xscaled = preprocessing.normalize(X)
        # derive dp-mean
        mu = np.mean(Xscaled, axis=0)
        bMu = np.sqrt(M) / (N * epsMu)
        prng = np.random.RandomState(seed=prng_seed)
        lapNoise = prng.laplace(scale=bMu, size=M)
        muPriv = mu + lapNoise
        # centering
        Xbar = Xscaled - muPriv
        # re-normalize
        Xbar = preprocessing.normalize(Xbar)
        return Xbar, muPriv

    def _ron_projection(self, Xbar, dim):
        (N, M) = Xbar.shape
        randProj = self._generate_rand_onproj(M)
        onProj = randProj[0:dim]  # take the rows
        Xred = np.inner(Xbar, onProj)
        return Xred, onProj

    def _reconstruction(self, Xproj, onProj):
        Xrecon = np.inner(Xproj, onProj.T)
        return Xrecon

    def _generate_rand_onproj(self, m):
        # generate random matrix
        randMat = np.random.uniform(size=(m, m))
        # QR factorization
        Q, R = scipy.linalg.qr(randMat)
        direction = Q
        return direction
