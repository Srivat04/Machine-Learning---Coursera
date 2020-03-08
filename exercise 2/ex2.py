import numpy as np
import matplotlib.pyplot as plt
from LogisticRegressor import Log_reg

data1 = []
f = open("ex2data1.txt", "r")
for line in f.readlines() :
	data1.append(line.split("\n")[0].split(","))

data1 = np.array(data1, dtype = 'float64')

f.close()

X = data1[:,:-1]
y = data1[:,-1]

model = Log_reg(data1, lamda = 0)
model.train()
theta = model.theta

pos = X[y == 1]
neg = X[y == 0]

plt.figure(figsize = (8,6))
plt.plot(pos[:,0],pos[:,1],'rx',label = "Class 1",markersize = 6)
plt.plot(neg[:,0],neg[:,1],'k+',label = "Class 2",markersize = 6)
plt.plot(X[:,0], (-theta[0]-theta[2]*X[:,0])/theta[1], 'g-', lw = 0.5, label = "Decision Boundary")
plt.title("Data")
plt.xlabel("X1")
plt.ylabel("Y1")
plt.legend()
plt.show()

p = model.predict(X)
print("Accuracy is {}".format(np.mean(p == y)*100))