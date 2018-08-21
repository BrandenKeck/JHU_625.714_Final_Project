######################################################################################
##### [TITLE] #####
##### By: Branden Keck, JHU 625.714: Intro. To Stochastic Differential Equations #####
######################################################################################
######################################################################################

# Nystrom approximation
def approximateNystrom(x, start, ny, nytol, n):
	
	# compute a minimum nystrom matrix size:
	check = start
	for i in range(start, n):
		if(np.array(x[start]).all != np.array(x[i]).all):
			check = i
			break;
	
	# check that Nystrom Segment is at least as large as the minimum
	if((start+ny)<check):
		print("ERROR: Current Nystrom block value is insufficient.  The value will be increased to the next minimum value.  This may make the simulation computationally insufficient.")
		end = check
	else:
		end = start + ny
		
	if(end>n): end = n

	# Computation of the Sample matrix A
	A = np.zeros([(end-start), (end-start)])
	for i in range(start, end):
		j = i
		while j<end:
			A[i-start,j-start] = np.exp(-1*np.linalg.norm(np.subtract(x[i],x[j]))/2*sig**2)
			if(i!=j):
				A[j-start,i-start] = A[i-start,j-start]
			j = j + 1
			
	A = tf.convert_to_tensor(A)
	Ainv = tf.matrix_inverse(A)
	
	# Computation of the interaction between the remaining points and the sample
	B = np.zeros([(n-end), (end-start)])
	for i in range(end, n):
		for j in range(start, end):
			B[i-end, j-start] = np.exp(-1*np.linalg.norm(np.subtract(x[i],x[j]))/2*sig**2)

	B = tf.convert_to_tensor(B)
	Btran = tf.transpose(B)
	
	# Fill in the gaps via estimation of C and the overall matrix W
	if((n-end-start) > nytol):
		print("Next recursion level reached...")
		C = approximateNystrom(x, end, ny, nytol, n)
	else:
		C = tf.matmul(B, tf.matmul(Ainv, Btran))
	
	W = tf.concat([tf.concat([A,Btran], 1), tf.concat([B,C], 1)], 0)
	return W

# Construction of the Diffusion Map via Nystrom Approximation
def diffusionMap(x, n, m, t, sig, ny, nytol):
		
	W = approximateNystrom(x, 0, ny, nytol, n)
	
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

def kMeans(DM, n, shape):
	
	# Begin with a k of 2 and step until an "elbow" is reached:
	# Add a tolerance for the point in which error changes by too little to continue
	# Start with an arbitrary change in error that is greater than the tolerance
	k = 2
	tol = 0.0001
	dE = 1
	prev_err = 1
	clusters = []
	while dE > tol or k < 5:
		clus = testK(k, DM, n, shape)
		clusters.append(clus[0])
		dE = np.abs(prev_err - clus[1])
		prev_err = clus[1]
		k = k + 1
		
		print("")
		print("Hey there, here's your improvement in error")
		print(dE)
		input()
		
	return [k, clusters[len(clusters)-2]]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def testK(k, DM, n, shape):
	
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
	tol = 0.0001
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
		
		if(loophold < 2):
			dE = error+1
		
		loopbreak = loopbreak + 1
		if(loopbreak > 100):
			dE = -1
		
	# Construction of test image
	l = []
	clusterMe = clusterMe.astype(int)
	for col in range(n):
		l.append(colors[clusterMe[col]])
	
	l = np.array(l, dtype='f').reshape(shape)
	
	# For testing purposes:
	print("")
	print(clusterMe)
	print(error)
	print(k)
	
	# Also for the testing phase
	plt.imshow(l)
	plt.show()
	input()
	
	return [clusterMe, error]
	
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
	file = '_data/Set0/1.png'
	
	# Adjustable model parameters
	sig = 0.001 # Scaling parameter for Diffusion Map kernel
	m = 2 # Number of eigenvalues to be included in Diffusion Map
	t = 0.2 # Diffusion Map time step
	Napp = 50 # Nystrom approximation segment length
	Ntol = 500 # Nystrom tolerance for final matrix multiplication
	
	# Gather the dataset and Nystrom approximation parameter
	[x, n, shape] = importData(file)
			
	# Update screen with current time
	print("")
	print("Time for Imports:")
	print(time.time() - start_me)
	
	# Compute a diffusion map from the data
	if(Napp<2): Napp = 2
	if(Napp>n): Napp = n
	if(Napp>Ntol): Ntol = Napp
	DM = diffusionMap(x, n, m, t, sig, Napp, Ntol)
	
	# Update screen with current time
	print("")
	print("Time for Diffusion Map Calculation:")
	print(time.time() - start_me)
	
	# Calculate "k" and clusters via k means
	clusters = kMeans(DM, n, shape)
	
	# Update screen with current time
	print("")
	print("Time for K Means Calculation:")
	print(time.time() - start_me)
	
	# Get data cluster centers
	