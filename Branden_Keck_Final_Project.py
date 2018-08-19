######################################################################################
##### [TITLE] #####
##### By: Branden Keck, JHU 625.714: Intro. To Stochastic Differential Equations #####
######################################################################################
######################################################################################


def diffusionMap(x, n, m, t, sig):

	# Construction of appropriate matrices for use in Diffusion Map algorithm
	W = []
	D = np.zeros([n, n])
	for i in range(0, n):
		dii = 0
		for j in range(0, n):
			dist = np.exp(-1*np.linalg.norm(np.subtract(x[i],x[j]))/2*sig**2)
			W.append(dist)
			D[i,j] = 0
			dii = dii + dist
		D[i,i] = dii
			
	# Finalization of diffusion matrix
	W = np.array(W)
	W = W.reshape(n, n)

	# Eigen solutions using numpy library
	invD = np.linalg.inv(D)
	A = np.matmul(invD, W)
	lam, evA = np.linalg.eig(A)

	# Final diffusion map computation
	DM = []
	for i in range(1,m+1):
		DM.append(lam[i]**t * np.array(evA[:,i]))
		
	DM = np.array(DM)
	return DM

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def kMeans(DM, n):
	
	# Begin with a k of 2 and step until an "elbow" is reached:
	# Add a tolerance for the point in which error changes by too little to continue
	# Start with an arbitrary change in error that is greater than the tolerance
	k = 2
	tol = 0.0001
	dE = 1
	prev_err = 1
	clusters = []
	while dE > tol:
		clus = testK(k, DM, n)
		clusters.append(clus[0])
		dE = np.abs(prev_err - clus[1])
		prev_err = clus[1]
		k = k + 1
		
		print("")
		print("Hey there, here's your improvement in error")
		print(dE)
		input()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def testK(k, DM, n):
	
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
		
		for i in range(k):
			points = [DM[:,j] for j in range(0,n) if clusterMe[j] == i]
			if(points==[]):
				scc[i] = DM.mean(axis=1)
			else:
				scc[i] = np.mean(points, axis=0)
		
		dE = np.abs(error - np.linalg.norm(np.subtract(scc, scc_old)))
		error = np.linalg.norm(np.subtract(scc, scc_old))
		print(str(dE))
		
		loopbreak = loopbreak + 1
		if(loopbreak > 100):
			dE = -1
		
	# Construction of test image
	l = []
	clusterMe = clusterMe.astype(int)
	for col in range(n):
		l.append(colors[clusterMe[col]])
	
	l = np.array(l, dtype='f').reshape([20, 20, 3])
	
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

def importData():
    
	# Get user-defined location
	print("")
	#lat = input("Enter a Latitude:  ")
	#lon = input("Enter a Longitude: ")
	
	# Get current datetime
	now = int(time.time())
	print(now)
	#start = 1369728000
	#end = 1369789200
	
	n = noaa.NOAA()
	observations = n.get_observations('15057','US',start='1369728000',end='1369789200')
	for observation in observations:
		print("")
		print(observation)
	
	# Data import
	#file = os.path.join('_test/20.png')
	#API_Key = "00c59abbcf91d81f6c2c00eeeca50ab4"
	#getMe = "http://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={start}&end={end}"
	#r = requests.get(getMe)
	#r.encoding = "ISO-8859-1"
	
	#lenna = io.imread(file)
	#shape = lenna.shape

	# Contruction of data vectors to be used in the Diffusion Map
	#x = []
	#for i in range(0, shape[0]):
	#	for j in range(0, shape[1]):
	#		x.append([i, j, lenna[i,j][0], lenna[i,j][1], lenna[i,j][2]])
	
	x=0		
	
	# ~~~~~~~~~~~~~ TEMPORARY FILE SAVE
	#np.savetxt("./weatherData.txt", condition)
	
	return x

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
	
if __name__ == "__main__":
	
	# ~~~~~~~~~~~~~~~ TEMPORARY FIX: COMPLEX WARNING ISSUES IGNORED
	import warnings
	warnings.filterwarnings('ignore')
	
	# Python Library Imports
	import os, requests, time
	import numpy as np
	from skimage import io
	from noaa_sdk import noaa
	from matplotlib import pyplot as plt
	from copy import deepcopy

	# Adjustable model parameters
	sig = 0.1; # Scaling parameter for Diffusion Map kernel
	m = 5; # Number of eigenvalues to be included in Diffusion Map
	t = 12; # Diffusion Map time step
	
	# Gather the dataset
	# ~~~~~~~ For now, use sample Image Data
	x = importData()
	exit()
	n = len(x)
	
	# Update screen with current time
	print(n)
	print("Imports Retrieved")
	
	input() #~~~~~~~~~~~~~~~~~~~~TEMPORARY BREAK
	quit()
	
	# Compute a diffusion map from the data
	DM = diffusionMap(x, n, m, t, sig)
	
	# Update screen with current time
	print("")
	print("Diffusion Map Calculated")
	
	# Calculate "k" and clusters via k means
	clusters = kMeans(DM, n)
	
	# Update screen with current time
	print("")
	print("K Means Calculated")
	
	# Update screen with current time
	print("")
	print("END PREDICTION")
	print("")