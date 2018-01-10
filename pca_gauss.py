#---------------------------
# pca_gauss.py
# Author: Thee Chanyaswad
#
# Version 1.0
#	- Initial implementation.
#---------------------------

import numpy as np

class PCA-Gauss:
	
	def __init__(self, algorithm='supervised', epsilonGauss=1.0, epsilonPca=1.0):
		self.algorithm = algorithm
		self.epsilonGauss = epsilonGauss
		self.epsilonPca = epsilonPca
	
	def dppca_fit(self, X, a=1.0, seed = 0):
		self._dppca = DPPCA(X, self.epsilonPca, a, seed)
	
	def dppca_transform(self, X, dimension):
		Xred = self._dppca.transform(X, dimension)
		return Xred
		
	def generate_dpdata(self, Xprivate, dimension, yPrivate = None, numberOfSamples=5000, a=1.0, seed=0):
		epsilon = self.epsilonGauss
		self.dppca_fit(Xprivate, a, seed)
		Xred = self.dppca_transform(Xprivate, dimension)
		
		if self.algorithm == 'supervised':
		
			(N,P) = Xred.shape	
			mu = np.zeros(P + 1)
			b = ((P**2)*(a**2) + 4*P*(a**2) + (a**2))/(N*epsilon)
	
			yReshaped = yPrivate.reshape(len(yPrivate),1)
			augMat = np.hstack((Xred,yReshaped))
			scatMat = np.inner(augMat.T,augMat.T)/N
			prng = np.random.RandomState(seed)
			lapNoise = prng.laplace(scale=b,size=(P+1,P+1))
			cov = scatMat + lapNoise
			
			prng = np.random.RandomState(seed)
			synthData = prng.multivariate_normal(mu,cov, numberOfSamples)
			synthX = synthData[:,0:-1]
			synthY = synthData[:,-1]
			
			return (synthX, synthY)
	
		elif algorithm == 'unsupervised':
			
			(N,P) = Xred.shape
			mu = np.zeros(P)
			b = (P*(a**2))/(N*epsilon)
			
			scatMat = np.inner(Xred.T,Xred.T)/N
			prng = np.random.RandomState(seed)
			lapNoise = prng.laplace(scale=b,size=(P,P))
			cov = scatMat + lapNoise
			
	
			prng = np.random.RandomState(seed)
			synthData = prng.multivariate_normal(mu,cov, numberOfSamples)
	
			return synthData
		
		
		elif algorithm == 'gmm':
			epsMu = 0.1 * epsilon
			epsCov = 0.9 * epsilon
			synX = np.array([])
			synY = np.array([])
	
			for lab in np.unique(yPrivate):
				idx = np.where(label == lab)
				x = Xred[idx]
				muL = np.mean(x,axis=0)
				(N,P) = x.shape
		
				XredBar = x - muL
		
				b = ((P**2)*(a**2))/(N*epsCov)
	
				scatMat = np.inner(XredBar.T,XredBar.T)/N
				prng = np.random.RandomState(seed)
				lapNoise = prng.laplace(scale=b,size=(P,P))
				cov = scatMat + lapNoise
				bMu = 2*P*a/(N*epsMu)
				lapNoise = prng.laplace(scale=bMu,size=P)
				muPriv = muL + lapNoise
		
				synthData = prng.multivariate_normal(muPriv,cov, numberOfSamples)
		
				if len(synX) == 0:
					synX = synthData
				else:
					synX = np.vstack((synX,synthData))
			
				synY = np.append(synY,lab*np.ones(numSamples))
	
			return (synX,synY)
	
		



class DPPCA:
	def __init__(self, X, epsilon, a=1.0, seed = 0):
		self.mean_ = np.mean(X, axis=0)
		Xbar = X - self.mean_
		(N,M) = X.shape
		
		b = ((M**2)*(a**2) + 4*M*(a**2) + (a**2))/(N*epsilon)
		
		R = np.inner(Xbar.T,Xbar.T)/N
		prng = np.random.RandomState(seed)
		lapNoise = prng.laplace(scale=b,size=(M,M))
		cov = R + lapNoise
		eigVal = np.linalg.eigvalsh(cov)
		if eigVal[0] < 0.0:
			cov = cov - 1.1*eigVal[0]*np.eye(M)
					
		U, S, V = np.linalg.svd(cov, full_matrices= True)
		self.allComponents = V
		self.components_ = V
		self.singVal = S
		self.U = U
		self.V = V
	
	def transform(self, X, dim):
		Xred = np.inner(self.components_[0:dim],X).T
		return Xred