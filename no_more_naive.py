##############################################################################################
####################### Spectral Clustering via Diffusion Maps: ##############################
##### Algorithm Optimization and Applications in Predictive Analysis of Weather Patterns #####
######### By: Branden Keck, JHU 625.714: Intro. To Stochastic Differential Equations #########
################################## 22 August 2018 ############################################
##############################################################################################

# Construction of the Diffusion Map via Nystrom Approximation
def diffusionMap(x, n, m, t, sig):
	
	nystroem = ka.Nystroem(kernel="rbf", n_components=n, gamma=sig**2)
	W = nystroem.fit_transform(x)
	#W = sk.rbf_kernel(x, Y=None, gamma=sig**2)
	
	#nystroem = ka.Nystroem(kernel="linear", n_components=n, gamma=sig**2)
	#W = nystroem.fit_transform(x)
	
	# Computation the diagonal matrix of row sums
	d = tf.reduce_sum(W, 1)
	Dinv = tf.matrix_inverse(tf.diag(d))
	
	# Creation of stochastic matrix M and its eigenfunctions
	M = tf.matmul(Dinv, W)
	M = tf.Session().run(M)
	lam, evA = np.linalg.eig(M)

	# Final diffusion map computation
	DM = []
	for i in range(1,m+1):
		DM.append(lam[i]**t * np.array(evA[:,i]))
	DM = np.transpose(np.array(DM))
	
	return DM



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def kMeans(k, DM, n, shape):
	
	colors = [[0,0,0],[20,20,20],[50,50,50],[100,100,100],[140,140,140],[180,180,180],[230,230,230]]
	
	km = clus.KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, tol=0.0001).fit(DM)
	clusterMe = km.labels_
		
	# Construction of test image
	l = []
	clusterMe = clusterMe.astype(int)
	print(clusterMe)
	for col in range(n):
		l.append(colors[clusterMe[col]])
	l = np.array(l, dtype='f').reshape(shape)
	
	# Also for the testing phase
	plt.imshow(l)
	plt.show()
	
	# return [clusterMe, error] # Come back to this?
	return clusterMe
	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def analyzeClusters(k, x, clus, shape):
	c = clus.reshape([shape[0], shape[1]])
	print(c)

	bg = c[0,0]
	counts = np.zeros(k)
	meanX = np.zeros(k)
	meanY = np.zeros(k)
	meanR = np.zeros(k)
	meanG = np.zeros(k)
	meanB = np.zeros(k)
	for i in range(0, shape[0]):
		for j in range(0, shape[1]):
			if(c[i,j]) != bg:
				counts[c[i,j]] = counts[c[i,j]] + 1 #Count
				meanX[c[i,j]] = meanX[c[i,j]] + i #Count
				meanY[c[i,j]] = meanY[c[i,j]] + j #Count
				
	for i in range(0,k):
		if i!=bg and counts[i]!=0:
			meanX[i] = meanX[i]/counts[i]
			meanY[i] = meanY[i]/counts[i]
			
	print(meanX)
	print(meanY)
	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def importData(file):

	# Data import
	file = os.path.join(file)
	lenna = io.imread(file)
	shape = lenna.shape

	# Contruction of data vectors to be used in the Diffusion Map
	x = []
	for i in range(0, shape[0]):
		for j in range(0, shape[1]):
			x.append([i, j, lenna[i,j][0], lenna[i,j][1], lenna[i,j][2]])
			
	return [x, len(x), shape]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def drawPredictions():
	x = 1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
	
if __name__ == "__main__":

	# Start time for benchmarking the code
	import time
	start_me = time.time()
	
	
	# Suppress warnings due to complex eigenvalues
	import warnings
	warnings.filterwarnings('ignore')
	
	
	# Python Library Imports
	import os
	import sklearn.metrics.pairwise as sk
	import sklearn.kernel_approximation as ka
	import sklearn.cluster as clus
	import tensorflow as tf
	import numpy as np
	from skimage import io
	from matplotlib import pyplot as plt
	from copy import deepcopy
	
	# Define the dataset
	files = ['_data/Set0/1.png', '_data/Set0/2.png', '_data/Set0/3.png', '_data/Set0/4.png', '_data/Set0/5.png']
	#files = ["_data/Set1minmin/1.png"]
	#files = ["_data/_test/10.png"]
	
	# Adjustable model parameters
	sig = 0.001 # Scaling parameter for Diffusion Map kernel
	m = 2 # Number of eigenvalues to be included in Diffusion Map
	t = 0.2 # Diffusion Map time step
	k = 3 # K-Means Clustering constant
	numP = 2 # Number of future prediction images
	
	counter = 1
	for file in files:
		print("Calculating for file #" + str(counter))
	
		# Gather the dataset and Nystrom approximation parameter
		[x, n, shape] = importData(file)
				
		# Update screen with current time
		print("")
		print("Time for Imports:")
		print(time.time() - start_me)
		
		# Compute a diffusion map from the data
		DM = diffusionMap(x, n, m, t, sig)
		
		# Update screen with current time
		print("")
		print("Time for Diffusion Map Calculation:")
		print(time.time() - start_me)
		
		
		# Calculate "k" and clusters via k means
		clusters = kMeans(k, DM, n, shape)
		
		# Update screen with current time
		print("")
		print("Time for K Means Calculation:")
		print(time.time() - start_me)
		
		# Get data cluster centers
		analyzeClusters(k, x, clusters, shape)
		
		counter = counter + 1
		
	drawPredictions()
	
	print("")
	print("-------------------")
	print("SIMULATION COMPLETE")
	