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
        (n, m) = X.shape
        if n_samples is None:
            n_samples = n

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
        (x_bar, mu_dp) = self._data_preprocessing(X, self.epsilon_mean, prng)
        (x_tilda, proj_matrix) = self._apply_ron_projection(x_bar, dimension, prng)

        (n, p) = x_tilda.shape
        noise_var = (2.0 * np.sqrt(p)) / (n * self.epsilon_cov)
        cov_matrix = np.inner(x_tilda.T, x_tilda.T) / n
        laplace_noise = prng.laplace(scale=noise_var, size=(p, p))
        cov_dp = cov_matrix + laplace_noise
        synth_data = prng.multivariate_normal(np.zeros(p), cov_dp, n_samples)
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
        (x_bar, mu_dp) = self._data_preprocessing(X, self.epsilon_mean, prng)
        (x_tilda, proj_matrix) = self._apply_ron_projection(x_bar, dimension, prng)

        (n, p) = x_tilda.shape
        noise_var = (2.0 * np.sqrt(p) + 4.0 * np.sqrt(p) * max_y + max_y ** 2) / (
            n * self.epsilon_cov
        )
        y_reshaped = y.reshape(len(y), 1)
        augmented_mat = np.hstack((x_tilda, y_reshaped))
        cov_matrix = np.inner(augmented_mat.T, augmented_mat.T) / n
        laplace_noise = prng.laplace(scale=noise_var, size=cov_matrix.shape)
        cov_dp = cov_matrix + laplace_noise

        synth_data = prng.multivariate_normal(np.zeros(p + 1), cov_dp, n_samples)
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
        syn_x = []
        syn_y = np.array([])
        for label in np.unique(y):
            idx = np.where(y == label)
            x_class = X[idx]
            (x_bar, mu_dp) = self._data_preprocessing(x_class, self.epsilon_mean, prng)
            (x_tilda, proj_matrix) = self._apply_ron_projection(x_bar, dimension, prng)

            (n, p) = x_tilda.shape
            noise_var = (2.0 * np.sqrt(p)) / (n * self.epsilon_cov)
            mu_dp_tilda = np.inner(mu_dp, proj_matrix)
            cov_matrix = np.inner(x_tilda.T, x_tilda.T) / n
            laplace_noise = prng.laplace(scale=noise_var, size=(p, p))
            cov_dp = cov_matrix + laplace_noise
            synth_data = prng.multivariate_normal(mu_dp_tilda, cov_dp, n_samples)
            if reconstruct:
                synth_data = self._reconstruction(synth_data, proj_matrix)
            
            syn_x.append(synth_data)
            syn_y = np.append(syn_y, label * np.ones(n_samples))
        
        syn_x = np.array(syn_x)
        return syn_x, syn_y

    @staticmethod
    def _data_preprocessing(X, epsilon_mean, prng=None):
        if prng is None:
            prng = np.random.RandomState()
        (n, m) = X.shape
        # pre-normalize
        x_norm = preprocessing.normalize(X)
        # derive dp-mean
        mu = np.mean(x_norm, axis=0)
        noise_var_mu = np.sqrt(m) / (n * epsilon_mean)
        laplace_noise = prng.laplace(scale=noise_var_mu, size=m)
        dp_mean = mu + laplace_noise
        # centering
        x_bar = x_norm - dp_mean
        # re-normalize
        x_bar = preprocessing.normalize(x_bar)
        return x_bar, dp_mean

    def _apply_ron_projection(self, x_bar, dimiension, prng=None):
        (n, m) = x_bar.shape
        full_projection_matrix = self._generate_ron_matrix(m, prng)
        ron_matrix = full_projection_matrix[0:dimension]  # take the rows
        x_tilda = np.inner(x_bar, ron_matrix)
        return x_tilda, ron_matrix

    def _reconstruction(self, x_projected, ron_matrix):
        x_reconstructed = np.inner(x_projected, ron_matrix.T)
        return x_reconstructed

    def _generate_ron_matrix(self, m, prng=None):
        if prng is None:
            prng = np.random.RandomState()
        # generate random matrix
        random_matrix = prng.uniform(size=(m, m))
        # QR factorization
        q_matrix, r_matrix = scipy.linalg.qr(random_matrix)
        ron_matrix = q_matrix
        return ron_matrix
