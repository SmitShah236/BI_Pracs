import pandas as pd 
import numpy as np 
import math
import operator
import matplotlib.pyplot as plt 

data = pd.read_csv('C:\BI Pracs\csv files\headbrain.csv')
print(data.head())

X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

mean_x = np.mean(X)
mean_y = np.mean(Y)

m = len(X)

numer = 0
denom = 0
for i in range(m):
  numer += (X[i] - mean_x) * (Y[i] - mean_y)
  denom += (X[i] - mean_x) ** 2
m = numer / denom
c = mean_y - (m * mean_x)

print (f'm = {m} \nc = {c}')

max_x = np.max(X) + 100
min_x = np.min(Y) - 100

x = np.linspace (min_x, max_x, 100)
y = c + m * x

plt.plot(x, y, color='#58b970', label='Regression Line')
plt.scatter(X, Y, c='#ef5423', label='data points')

plt.xlabel('Head Size in cm')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

ss_t = 0 
ss_r = 0 

for i in range(int(val_count)): 
  y_pred = c + m * X[i]
  ss_t += (Y[i] - mean_y) ** 2
  ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)

print(r2)