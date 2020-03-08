import numpy as np
import matplotlib.pyplot as plt
from LogisticRegressor import Log_reg
from FeatureMap import map_feature
from utils import plotDecisionBoundary


data2 = []
f = open("ex2data2.txt", "r")
for line in f.readlines() :
	data2.append(line.split("\n")[0].split(","))

data2 = np.array(data2, dtype = 'float64')

f.close()

X = data2[:,:-1]
y = data2[:,-1]

pos = X[y == 1]
neg = X[y == 0]

plt.figure(figsize = (8,6))
plt.plot(pos[:,0],pos[:,1],'rx',label = "Class 1",markersize = 6)
plt.plot(neg[:,0],neg[:,1],'k+',label = "Class 2",markersize = 6)
plt.title("Data")
plt.xlabel("X1")
plt.ylabel("Y1")
plt.legend()
plt.show()

model = Log_reg(data2, poly_features = 6, lamda = 1)
model.train()
print(model.theta)
plotDecisionBoundary(plotData, theta, map_feature(X), y)

p = model.predict(X)
print("Accuracy is {}%".format(np.mean(p == y)*100))
