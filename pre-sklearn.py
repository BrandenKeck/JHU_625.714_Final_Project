##############################################################################################
####################### Spectral Clustering via Diffusion Maps: ##############################
##### Algorithm Optimization and Applications in Predictive Analysis of Weather Patterns #####
######### By: Branden Keck, JHU 625.714: Intro. To Stochastic Differential Equations #########
################################## 22 August 2018 ############################################
##############################################################################################

# recursive Nystrom approximation algorithm
def approximateNystrom(x, n, A, step):
	print(tf.Session().run(A))
	input() 
	
	stop = False
	
	# Compute the inverse of A
	Ainv = tf.matrix_inverse(A)
	
	# Determin the length of the recursive "B"
	blf = 0.5
	if((n-step)<500):
		bl = (n-step)
		stop = True
	else:
		bl = int((n-step)*blf)
		
	# Computation of the interaction between the remaining points and the sample (B and Btran)
	B = np.zeros([bl, step])
	for i in range(0, bl):
		for j in range(0, step):
			B[i, j] = np.exp(-1*np.linalg.norm(np.subtract(x[i+step],x[j]))/2*sig**2)
	B = tf.convert_to_tensor(B)
	Btran = tf.transpose(B)
	step = step+bl
	print(tf.Session().run(B))
	input() 
	
	# Estimation of non-explicitly computed points
	C = tf.matmul(B, tf.matmul(Ainv, Btran))
	print(tf.Session().run(C))
	input() 
	
	# New A - Combine matrix segments
	newA = tf.concat([tf.concat([A,Btran], 1), tf.concat([B,C], 1)], 0)
	print(tf.Session().run(C))
	input() 
	
	if stop:
		return newA
	else:
		approximateNystrom(x, n, newA, step)
	
	

# Construction of the Diffusion Map via Nystrom Approximation
def diffusionMap(x, n, m, t, sig, ny):
	ny = n
	# Computation of the Sample matrix A
	A = np.zeros([ny, ny])
	for i in range(0, ny):
		j = i
		while j<ny:
			A[i,j] = np.exp(-1*np.linalg.norm(np.subtract(x[i],x[j]))/2*sig**2)
			if(i!=j):
				A[j,i] = A[i,j]
			j = j + 1
	A = tf.convert_to_tensor(A)
	
	# Recursive Nystrom approximation calculating only the "B" and "C" matrices
	#W = approximateNystrom(x, n, A, ny)
	W = A
	
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
	DM = np.array(DM)
	
	return DM


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def kMeans(k, DM, n, shape):
	
	# Initialize centroids for spectral clustering
	scc = np.zeros([k, m])
	for i in range(0,k):
		for j in range(0,m):
			# ~~~~~~~~~~~~~ THERE'S AN IMAGINARY NUMBER WARNING HERE
			scc[i, j] = np.random.uniform(np.min(DM[:,:]), np.max(DM[:,:]))

	# Quick Recoloring scheme - REMOVE
	colors = [[0,0,0],[20,20,20],[50,50,50],[100,100,100],[140,140,140],[180,180,180],[230,230,230]]
			
	# Movement of centroids via spectral clustering algorithm		
	scc_old = np.zeros(scc.shape)
	clusterMe = np.zeros(n)
	error = np.linalg.norm(np.subtract(scc, scc_old))

	# Minimization 
	dE = 10
	tol = 0.0000001
	loopbreak = 0
	while dE > tol:

		for i in range(0,n):
			distances = scc - DM[:,i]
			distances = np.sum(np.abs(distances)**2,axis=-1)**(1./2)
			clusterMe[i] = int(np.argmin(np.array(distances)))

		scc_old = deepcopy(scc)
		
		loophold = 0
		for i in range(k):
			points = [DM[:,j] for j in range(0,n) if clusterMe[j] == i]
			if(points==[]):
				scc[i] = DM.mean(axis=1)
			else:
				scc[i] = np.mean(points, axis=0)
				loophold = loophold + 1
		
		dE = np.abs(error - np.linalg.norm(np.subtract(scc, scc_old)))
		error = np.linalg.norm(np.subtract(scc, scc_old))
		
		# Allow only one empty cluster
		if(loophold < k-1):
			dE = error+1
		
		# Break loop given no solution
		loopbreak = loopbreak + 1
		if(loopbreak > 1000):
			dE = -1
		
	# Construction of test image
	l = []
	clusterMe = clusterMe.astype(int)
	for col in range(n):
		l.append(colors[clusterMe[col]])
	l = np.array(l, dtype='f').reshape(shape)
	
	# Also for the testing phase
	plt.imshow(l)
	plt.show()
	input()
	
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
	
if __name__ == "__main__":

	# Start time for benchmarking the code
	import time
	start_me = time.time()
	
	
	# Suppress warnings due to complex eigenvalues
	import warnings
	warnings.filterwarnings('ignore')
	
	
	# Python Library Imports
	import os
	import numpy as np
	import tensorflow as tf
	from skimage import io
	from matplotlib import pyplot as plt
	from copy import deepcopy
	
	# Define the dataset
	#files = ['_data/Set0/1.png', '_data/Set0/2.png', '_data/Set0/3.png', '_data/Set0/4.png', '_data/Set0/5.png']
	#files = ["_data/Set0/1.png"]
	files = ["_data/_test/10.png"]
	
	# Adjustable model parameters
	sig = 0.001 # Scaling parameter for Diffusion Map kernel
	m = 2 # Number of eigenvalues to be included in Diffusion Map
	t = 0.2 # Diffusion Map time step
	k = 4 # K-Means Clustering constant
	nyc = 31 # Nystrom constant
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
		if(nyc>n):nyc=n
		if(nyc<2):nyc=2
		DM = diffusionMap(x, n, m, t, sig, nyc)
		
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
	