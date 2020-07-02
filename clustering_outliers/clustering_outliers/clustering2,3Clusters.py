#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from sklearn.metrics import confusion_matrix
import csv

np.random.seed(0)
X, y = sklearn.datasets.make_moons(1000, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)


def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    

np.random.seed(0)
Xtest, ytest = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(Xtest[:,0], Xtest[:,1], s=40, c=ytest, cmap=plt.cm.Spectral)



with open('data_clusters2.csv', mode='w', newline='') as data_clusters2:
    
    fieldnames = ['count', 'x', 'y', 'class']
    writer = csv.DictWriter(data_clusters2, fieldnames=fieldnames, delimiter=",")
    writer.writeheader()
    for i in range(len(Xtest)):
        writer.writerow({"count": i, "x" : Xtest[i][0], "y" : Xtest[i][1], "class" : ytest[i]})
        

with open('data_clusters3.csv', mode='w', newline='') as data_clusters3:
    
    fieldnames = ['count', 'x', 'y', 'class']
    writer = csv.DictWriter(data_clusters3, fieldnames=fieldnames, delimiter=",")
    writer.writeheader()
    for i in range(len(Xtest)):
        writer.writerow({"count": i, "x" : Xtest3[i][0], "y" : Xtest3[i][1], "class" : ytest3[i]})

