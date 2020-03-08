import numpy as np

def map_feature (X, degree = 6) :
	
	flag = 1
	for x in X :
		feature = np.array([])
		for i in range(degree+1) :
			for j in range(degree+1-i) :
				feature = np.append(feature, x[0]**i*x[1]**j)
		if flag :
			features = feature
			features.shape = (1,features.shape[0])
			flag = 0
		else :
			features = np.append(features, [feature], axis = 0)

	return features

