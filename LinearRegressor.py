import numpy as np

class lin_reg :
	
	def __init__(self, train_data) :
		self.train_data = train_data
		self.X = np.ones(train_data.shape)
		self.X[:,1:] = train_data[:,:-1]
		self.y = train_data[:,-1]
		self.m = len(self.y)
		self.theta = np.random.randn((self.train_data.shape[1]))
		self.cost = []

	def predict(self, test) :

		return [self.theta[0] + self.theta[1:].dot(x) for x in test]
	
	def gradient(self, X, y, theta, m) :
		
		return (1/m)*((theta.dot(X.T) - y.T).dot(X))

	def train (self, alpha = 1e-7, iter = 1000) :
		print(self.theta)
		for i in range(iter) :
			self.theta = self.theta - alpha*self.gradient(self.X, self.y, self.theta, self.m)
			self.cost.append([i, self.loss()])

	def loss(self) :

		diff = self.X.dot(self.theta.T) - self.y
		loss = 1/(2*self.m)*diff.dot(diff.T)
		
		return loss
