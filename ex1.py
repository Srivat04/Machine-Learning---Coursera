import numpy as np
import matplotlib.pyplot as plt
import LinearRegressor 

data1 = []
f = open("ex1data1.txt", "r")
for line in f.readlines() :
	data1.append(line.split("\n")[0].split(","))

data1 = np.array(data1, dtype = 'float64')

f.close()

plt.figure(figsize = (8,6))
plt.plot(data1[:,0], data1[:,1], 'rx', markersize = 3.5)
plt.title("Scatterplot of the Data", fontsize = 15)
plt.xlabel("Population of City in 10,000s' ")
plt.ylabel("Profit in $10,000s'")
plt.show()

model = LinearRegressor.lin_reg(data1)
model.train(iter = 1000, alpha = 0.01)
cost = np.array(model.cost)

plt.figure(figsize = (8,6))
plt.plot(data1[:,0], data1[:,1], 'rx', markersize = 3.5)
plt.plot(data1[:,0], model.predict(data1[:,0]), 'g-', markersize = 3.5)
plt.title("Scatterplot of the Data", fontsize = 15)
plt.xlabel("Population of City in 10,000s' ")
plt.ylabel("Profit in $10,000s'")
plt.show()

plt.figure(figsize = (8,6))
plt.plot(cost[:,0], cost[:,1], 'r-', markersize = 3.5)
plt.title("Cost function vs Iteration", fontsize = 15)
plt.xlabel("Nuumber of Iterations ")
plt.ylabel("Cost value")
plt.show()