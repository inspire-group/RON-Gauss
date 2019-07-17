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


import numpy as np
import scipy
from sklearn import preprocessing


class RONGauss:
    def __init__(self, algorithm="supervised", epsilonMean=1.0, epsilonCov=1.0):
        self.algorithm = algorithm
        self.epsilonMean = epsilonMean
        self.epsilonCov = epsilonCov

    def generate_dpdata(
        self,
        X,
        dimension,
        y=None,
        numSamples=None,
        maxY=None,
        reconstruct=False,
        meanAdjusted=False,
        seed=7,
    ):
        prng = np.random.RandomState(seed)
        if self.algorithm == "unsupervised":
            (Xbar, muPriv) = self._data_preprocessing(X, self.epsilonMean)
            (Xred, onProj) = self._ron_projection(Xbar, dimension)

            (N, P) = Xred.shape
            b = (2.0 * np.sqrt(P)) / (N * self.epsilonCov)
            if numSamples is None:
                numSam = N
            else:
                numSam = numSamples
            scatMat = np.inner(Xred.T, Xred.T) / N
            lapNoise = np.random.laplace(scale=b, size=(P, P))
            dpCov = scatMat + lapNoise
            synthData = prng.multivariate_normal(np.zeros(P), dpCov, numSam)
            dpX = synthData
            dpY = None
            if reconstruct:
                dpX = self._reconstruction(dpX, onProj)
            else:
                muPriv = np.inner(muPriv, onProj)

            if meanAdjusted:
                dpX = dpX + muPriv

        elif self.algorithm == "supervised":
            (Xbar, muPriv) = self._data_preprocessing(X, self.epsilonMean)
            (Xred, onProj) = self._ron_projection(Xbar, dimension)

            (N, P) = Xred.shape
            b = (2.0 * np.sqrt(P) + 4.0 * np.sqrt(P) * maxY + maxY ** 2) / (
                N * self.epsilonCov
            )
            if numSamples is None:
                numSam = N
            else:
                numSam = numSamples
            yReshaped = y.reshape(len(y), 1)
            augMat = np.hstack((Xred, yReshaped))
            scatMat = np.inner(augMat.T, augMat.T) / N
            lapNoise = np.random.laplace(scale=b, size=(P + 1, P + 1))
            dpCov = scatMat + lapNoise

            synthData = prng.multivariate_normal(np.zeros(P + 1), dpCov, numSam)
            dpX = synthData[:, 0:-1]
            dpY = synthData[:, -1]
            if reconstruct:
                dpX = self._reconstruction(dpX, onProj)
            else:
                muPriv = np.inner(muPriv, onProj)

            if meanAdjusted:
                dpX = dpX + muPriv

        elif self.algorithm == "gmm":
            synX = np.array([])
            synY = np.array([])
            for lab in np.unique(y):
                idx = np.where(y == lab)
                xClass = X[idx]
                (Xbar, muPriv) = self._data_preprocessing(xClass, self.epsilonMean)
                (Xred, onProj) = self._ron_projection(Xbar, dimension)

                (N, P) = Xred.shape
                b = (2.0 * np.sqrt(P)) / (N * self.epsilonCov)
                if numSamples is None:
                    numSam = N
                else:
                    numSam = numSamples
                muRed = np.inner(muPriv, onProj)
                scatMat = np.inner(Xred.T, Xred.T) / N
                lapNoise = np.random.laplace(scale=b, size=(P, P))
                dpCov = scatMat + lapNoise
                synthData = prng.multivariate_normal(muRed, dpCov, numSam)
                if reconstruct:
                    synthData = self._reconstruction(synthData, onProj)
                if len(synX) == 0:
                    synX = synthData
                else:
                    synX = np.vstack((synX, synthData))

                synY = np.append(synY, lab * np.ones(numSam))
            dpX = synX
            dpY = synY

        return (dpX, dpY)

    def _data_preprocessing(self, X, epsMu):
        (N, M) = X.shape
        # pre-normalize
        Xscaled = preprocessing.normalize(X)
        # derive dp-mean
        mu = np.mean(Xscaled, axis=0)
        bMu = np.sqrt(M) / (N * epsMu)
        lapNoise = np.random.laplace(scale=bMu, size=M)
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
