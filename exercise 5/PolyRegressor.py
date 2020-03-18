import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize

class lin_reg2D :

	def __init__(self, lamda = 0) :
		self.lamda = lamda
		self.theta = np.ones((1,2))

	def hypothesis(self, theta, X) :

		return np.reshape(np.append(np.ones((len(X), 1)), X, axis = 1)@theta.T, (len(X),1))

	def cost(self,theta , X, y) :
		
		m = len(X)
		
		cost = 1/(2*m)*(norm(self.hypothesis(theta, X) - y, ord = "fro")**2 + self.lamda*theta@theta.T)
		grad = 1/m*(np.sum((self.hypothesis(theta, X)-y)*np.append(np.ones((len(X),1)), X, axis = 1), axis = 0) + self.lamda*np.append(0, theta[1:]))

		return cost, grad

	def train(self, X, y) :

		cost_function = lambda theta : self.cost(theta, X, y)
		theta_initial = self.theta
		
		options = {"maxiter":50}
		result = minimize(cost_function, theta_initial, jac = True, method = 'CG', options = options)

		self.theta = result.x

	def predict(self, X) :

		return np.reshape(np.append(np.ones((len(X), 1)), X, axis = 1)@self.theta.T, (len(X),1))

	def loss(self, X, y) :

		cost, _ = self.cost(self.theta, X, y)

		return cost


class poly_reg2D :

	def __init__(self, lamda = 100, degree = 4) :
		self.lamda = lamda
		self.d = degree
		self.theta = np.ones((1,degree + 1))

	def features(self, X) :

		features = np.ones((len(X),1))

		for i in range(self.d) :
			features = np.append(features, X**(i+1), axis = 1)

		return features

	def hypothesis(self, theta, X) :

		Xf = self.features(X)
		return np.reshape(Xf@theta.T, (len(X),1))

	def cost(self,theta , X, y) :
		
		m = len(X)
		Xf = self.features(X)
		cost = 1/(2*m)*(norm(self.hypothesis(theta, X) - y, ord = "fro")**2 + self.lamda*theta@theta.T)
		grad = 1/m*(np.sum((self.hypothesis(theta, X)-y)*Xf, axis = 0) + self.lamda*np.append(0, theta[1:]))

		return cost, grad

	def train(self, X, y) :

		cost_function = lambda theta : self.cost(theta, X, y)
		theta_initial = self.theta
		
		options = {"maxiter":500}
		result = minimize(cost_function, theta_initial, jac = True, method = 'CG', options = options)

		self.theta = result.x

	def predict(self, X) :

		Xf = self.features(X)
		return np.reshape(Xf@self.theta.T, (len(X),1))

	def loss(self, X, y) :

		cost, _ = self.cost(self.theta, X, y)

		return cost
