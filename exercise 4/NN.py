import numpy as np
from scipy.io import loadmat
from numpy.linalg import norm
from scipy.optimize import minimize

class nn_mnist :

	def __init__(self, inp_shape = 400, out_shape = 10, hidden_units = 25, lamda = 1, load_weights = False) :
		
		self.i = inp_shape
		self.o = out_shape
		self.h = hidden_units
		ep1, ep2 = (6/(hidden_units+inp_shape+1))**0.5, (6/(hidden_units+out_shape+1))**0.5 
		if load_weights :
			self.W1 = loadmat("ex4weights.mat")["Theta1"]
			self.W2 = loadmat("ex4weights.mat")["Theta2"]
		else :
			self.W1 = np.random.uniform(size = (hidden_units,inp_shape+1))*2*ep1 - ep1
			self.W2 = np.random.uniform(size = (out_shape,hidden_units+1))*2*ep2 - ep2

		self.lamda = lamda
		self.a0 = self.a1 = self.a2 = 0
		self.z1 = self.z2 = 0
		self.del1 = self.del2 = 0
		self.delta1 = self.delta2 = 0

	def sigmoid(self, x) :

		return 1/(1+np.exp(-x))

	def sig_der(self, x) :

		return self.sigmoid(x)*(1-self.sigmoid(x))

	def transformer(self, y) :

		y = np.array([y == j for j in range(1, self.o+1)], dtype = 'float64')[:,:,0]
		
		return y

	def regularizer(self, m) :

		return self.lamda/(2*m)*(norm(self.W1[:,1:], 'fro')**2+norm(self.W2[:,1 :], 'fro')**2)

	def cost_function (self,nn_params, X, y) :

		m = len(X)
		n = self.o
		y = self.transformer(y)
		h = self.hypothesis(X)

		self.W1 = nn_params[:self.h*(self.i+1)].reshape(self.W1.shape)
		self.W2 = nn_params[self.h*(self.i+1):].reshape(self.W2.shape)
		
		self.back_prop(X,y)
		grad = np.concatenate([self.delta1.ravel(), self.delta2.ravel()])

		J = np.sum([1/m*(-y[j]@np.log(h[j]+1e-25) - (1-y[j])@np.log(1-h[j]+1e-25)) for j in range(n)])+ self.regularizer(m)

		return J, grad

	def hypothesis(self, X) :

		h = X

		h = self.sigmoid(np.append(np.ones((len(h), 1)),h, axis = 1)@self.W1.T)
		h = self.sigmoid(np.append(np.ones((len(h), 1)),h, axis = 1)@self.W2.T)

		h = h.astype('float64')
		return h.T

	def feed_forward(self, x) :

		self.a0 = x
		self.a0 = np.append(1,self.a0)
		self.z1 = np.append(0,self.W1@self.a0)
		self.a1 = self.sigmoid(self.z1)
		self.z2 = self.W2@self.a1
		self.a2 = self.sigmoid(self.z2)	


	def back_prop(self, X, y) :

		y = y.T
		for i in range(len(X)) :

			self.feed_forward(X[i])
			self.del2 = self.a2 - y[i]
			self.del1 = self.W2.T@self.del2*self.sig_der(self.z1)
			self.del1 = self.del1[1:]

			self.delta1 += self.del1.reshape((len(self.del1), 1))@self.a0.reshape((1,len(self.a0)))
			self.delta2 += self.del2.reshape((len(self.del2), 1))@self.a1.reshape((1,len(self.a1)))


		self.delta1[:,1:] += self.lamda*self.W1[:,1:]
		self.delta2[:,1:] += self.lamda*self.W2[:,1:]

		self.delta1 /= len(X)
		self.delta2 /= len(X)

	def train(self, X, y) :

		initial_params = np.concatenate([self.W1.ravel(), self.W2.ravel()])
		costfunction = lambda w : self.cost_function(w, X, y)
		options = {"maxiter":100}
		result = minimize(costfunction, initial_params, jac = True, method = 'CG', options = options)

		optimal_params = result.x
		self.W1 = optimal_params[:self.h*(self.i+1)].reshape(self.W1.shape)
		self.W2 = optimal_params[self.h*(self.i+1):].reshape(self.W2.shape)

	def predict(self, X) :

		h = X

		h = self.sigmoid(np.append(np.ones((len(h), 1)),h, axis = 1)@self.W1.T)
		h = self.sigmoid(np.append(np.ones((len(h), 1)),h, axis = 1)@self.W2.T)

		h = h.astype('float64')
		h = np.argmax(h, axis = 1) + 1
		return h