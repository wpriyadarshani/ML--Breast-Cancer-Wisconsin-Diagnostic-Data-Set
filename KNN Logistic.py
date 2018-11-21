#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
import math
import csv
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier     #KNN
from sklearn.model_selection import KFold # import KFold
from sklearn.linear_model import LogisticRegression    #Logistic Regression
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt



# In[60]:


def loadData():
    # load  data
    total = pd.read_csv('data.csv')
    total_data = total.values
    np.random.shuffle(total_data)

    total_y = total_data[:,1]
    total_X = total_data[:,2:-1]


    #malignant 1 else 0
    for i in range(len(total_y)):
        if total_y[i] == 'M':
            total_y[i] = 1
        else:
            total_y[i] = 0

    # print(total_X)

    return total_X, total_y

total_X, total_y = loadData()


# In[61]:


kf = KFold(n_splits=10) # Define the split - into 10 folds 
kf.get_n_splits(total_X) # returns the number of splitting iterations in the cross-validator
print(total_X)
print(total_y)
total_X.shape
total_y.shape


# In[62]:


def KNN():
    accuracy = []
    kf = KFold(n_splits=10)
    for training_id, test_id in kf.split(total_X):
        train_x, test_x = total_X[training_id], total_X[test_id]
        train_y, test_y = total_y[training_id], total_y[test_id]

        train_y = train_y.astype('int') 
        knn = KNeighborsClassifier(13)
        knn.fit(train_x, train_y)

        output = knn.predict(test_x)
        correct = sum(np.array(output == test_y))

        # print(correct)

        accuracy.append((correct* 1.0)/ len(output))


    print("average accuracy KNN", (sum(accuracy)/ 10.0)*100)

KNN()


# In[63]:


def LogisticReg():
    accuracy = []
    kf = KFold(n_splits=10)
    for training_id, test_id in kf.split(total_X):
        train_x, test_x = total_X[training_id], total_X[test_id]
        train_y, test_y = total_y[training_id], total_y[test_id]

        train_y = train_y.astype('int') 
        log_r = LogisticRegression(solver='lbfgs')
        log_r.fit(train_x, train_y)

        output = log_r.predict(test_x)
        correct = sum(np.array(output == test_y))

        # print(correct)

        accuracy.append((correct* 1.0)/ len(output))


    print("average accuracy Logistic Regression", (sum(accuracy)/ 10.0)*100)

LogisticReg()







