import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import LinearRegressor 

data2 = []
f = open("ex1data2.txt", "r")
for line in f.readlines() :
	data2.append(line.split("\n")[0].split(","))

data2 = np.array(data2, dtype = 'float64')
f.close()

fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data2[:,0], data2[:,1], data2[:,2], c ='r', marker = 'o')
plt.show()


model = LinearRegressor.lin_reg(data2)
model.train(alpha = 1e-7, iter = 50)
cost = np.array(model.cost)

fig2 = plt.figure(figsize = (8,6))
ax2 = fig2.add_subplot(111, projection='3d')

ax2.scatter(data2[:,0], data2[:,1], model.predict(data2[:,:2]), c = 'g', marker = 'o')
ax2.scatter(data2[:,0], data2[:,1], data2[:,2], c ='r', marker = 'o')

plt.show()

plt.figure(figsize = (8,6))
plt.plot(cost[:,0], cost[:,1], 'r-', markersize = 3.5)
plt.title("Cost function vs Iteration", fontsize = 15)
plt.xlabel("Nuumber of Iterations ")
plt.ylabel("Cost value")
plt.show()