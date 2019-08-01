# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:55:24 2019

@author: HananeTech
"""

import numpy as np
import matplotlib.pyplot as plt


def get_data():
    # Nummbers of observation per class
    obs_per_class = 80
    
    X1 = np.random.randn(obs_per_class, 2) + np.array([-2, -2])
    X2 = np.random.randn(obs_per_class, 2) + np.array([2, 2])
    X = np.concatenate((X1, X2))
    y = np.concatenate((np.zeros(obs_per_class), np.ones(obs_per_class)))
    return X, y

def sigmoid(X, B, bias):
    """The sigmoid function"""
    return 1/(1+np.exp(-bias - (B.dot(X))))

def logistic_regression_sgd(X, y, epochs, l_rate):
    n = len(X[0]) #Number of features
    B = np.zeros(n)
    bias = 0
    for epoch in range(epochs):
        err = np.zeros(y.shape)
        for i in range(len(X)):
            predict = sigmoid(X[i, :], B, bias)
            err[i] = (predict - y[i])**2
            B = B + (l_rate*(y[i]-predict)*predict*(1-predict)*X[i])
            bias = bias + (l_rate*(y[i]-predict)*predict*(1-predict))
        print("Current error is:", np.mean(err))
    return B, bias

def to_predict(x, B, bias):
    return  np.round(sigmoid(x, B, bias))


X, y = get_data()

epochs = 10     #Iterations number
l_rate = 0.3    #Learning rate
N = len(X)      #Observations number
n = len(X[0])   #Features number

x=np.array([[2, 1]])    #Data to predict after the training process

#The training stage
B_final, bias_final = logistic_regression_sgd(X, y, epochs, l_rate)

#Print the learned coefficients
print(B_final,"\n", bias_final)
print("_________X1_______________X2_______________Prediction_________")
for i in range(N):
    print(X[i, 0],"_", X[i, 1], ":", sigmoid(X[i], B_final, bias_final))
print(sigmoid(x[0], B_final, bias_final))

#Prediction of a new data x
prediction = to_predict(x[0], B_final, bias_final)
print(prediction)

plt.scatter(X[:, 0], X[:, 1], s=30, c=y)

#Draw the line that separates the 2 classes
x_axis = np.linspace(-6, 5, 2)
y_axis = -(B_final[0]/B_final[1])*x_axis - (bias_final/B_final[1])
plt.plot(x_axis, y_axis)