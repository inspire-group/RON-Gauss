#---------------------------
# ron_gauss.py
# Author: Thee Chanyaswad
#
# Version 1.0
#	- Initial implementation.
#---------------------------

import numpy as np
import scipy

class RON-Gauss:
	
	def __init__(self, algorithm='supervised', epsilonMean=1.0, epsilonCov=1.0):
		self.algorithm = algorithm
		self.epsilonMean = epsilonMean
		self.epsilonCov = epsilonCov
	
	
	def generate_dpdata(self, X, dimension, y=None, maxY=None):
		if self.algorithm == 'unsupervised':
			(Xbar,_) = self.data_preprocessing(X,self.epsilonMean)
			(Xred,_) = self.ron_projection(Xbar,dimension)
			
			(N,P) = Xred.shape
			b = (2.*np.sqrt(P))/(N*self.epsilonCov)
			
			scatMat = np.inner(Xred.T,Xred.T)/N
			lapNoise = np.random.laplace(scale=b,size=(P,P))
			dpCov = scatMat + lapNoise
			synthData = prng.multivariate_normal(np.zeros(P),dpCov, N)
			dpX = synthData
			dpY = None
		
		elif self.algorithm == 'supervised':
			(Xbar,_) = self.data_preprocessing(X,self.epsilonMean)
			(Xred,_) = self.ron_projection(Xbar,dimension)
			
			(N,P) = Xred.shape
			b = (2.*np.sqrt(P) + 4.*np.sqrt(P)*maxY + a**2)/(N*self.epsilonCov)
			
			yReshaped = y.reshape(len(y),1)
			augMat = np.hstack((Xred,yReshaped))
			scatMat = np.inner(augMat.T,augMat.T)/N
			lapNoise = np.random.laplace(scale=b,size=(P+1,P+1))
			dpCov = scatMat + lapNoise
	
			synthData = prng.multivariate_normal(np.zeros(P + 1), dpCov, N)
			dpX = synthData[:,0:-1]
			dpY = synthData[:,-1]
		
		elif self.algorithm == 'gmm':
			synX = np.array([])
			synY = np.array([])
			for lab in np.unique(y):
				idx = np.where(y == lab)
				xClass = X[idx]
				(Xbar, muPriv) = self.data_preprocessing(xClass,self.epsilonMean)
				(Xred, onProj) = self.ron_projection(Xbar,dimension)
				
				(N,P) = Xred.shape
				b = (2.*np.sqrt(P))/(N*self.epsilonCov)
				muRed = np.inner(muPriv,onProj)
				scatMat = np.inner(Xred.T,Xred.T)/N
				lapNoise = np.random.laplace(scale=b,size=(P,P))
				dpCov = scatMat + lapNoise
				synthData = prng.multivariate_normal(muRed,dpCov, N)
				if len(synX) == 0:
					synX = synthData
				else:
					synX = np.vstack((synX,synthData))
				
				synY = np.append(synY,lab*np.ones(N))
				dpX = synX
				dpY = synY
			
		return (dpX,dpY)
		
	
	
	def data_preprocessing(self, X, epsMu):
		(N,M) = X.shape
		#pre-normalize
		Xscaled = preprocessing.normalize(X)
		#derive dp-mean
		mu = np.mean(Xscaled,axis=0)
		bMu = np.sqrt(M)/(N*epsMu)
		lapNoise = np.random.laplace(scale=bMu,size=M)
		muPriv = mu + lapNoise
		# centering
		Xbar = Xscaled - muPriv
		#re-normalize
		Xbar = preprocessing.normalize(Xbar)
		return Xbar, muPriv
	
	def ron_projection(self,Xbar,dim):
		(N,M) = Xbar.shape
		randProj = self.generate_rand_onproj(M)
		onProj = randProj[0:dim]
		Xred = np.inner(Xbar,onProj)
		return Xred, onProj
		
	
	def generate_rand_onproj(self,m):
		#generate random matrix
		randMat = np.random.uniform(size=(m,m))
		# QR factorization
		Q, R = scipy.linalg.qr(randMat)
		direction = Q
		return direction
		