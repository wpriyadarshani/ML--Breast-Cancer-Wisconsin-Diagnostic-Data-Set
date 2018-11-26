#!/usr/bin/env python
# coding: utf-8

# In[130]:


import pandas as pd
import numpy as np
import math
import csv
from sklearn import metrics
from sklearn.model_selection import KFold # import KFold
from sklearn.neighbors import KNeighborsClassifier     #KNN
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression    #Logistic Regression
import matplotlib.pylab as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


# In[131]:


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


# In[153]:


def KNN():
    
    kf = KFold(n_splits=10)
        # predict probabilities
        
      
    
    #try KNN for diffrent k nearest neighbor from 1 to 15
    neighbors_setting = range(1,16)
    for n_neighbors in neighbors_setting:
        print('k nearest neighbor=',n_neighbors)
        accuracy=[]
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        for training_id, test_id in kf.split(total_X):
            train_x, test_x = total_X[training_id], total_X[test_id]
            train_y, test_y = total_y[training_id], total_y[test_id]

            train_y=train_y.astype('int')
            knn.fit(train_x,train_y)
            output = knn.predict(test_x)
            correct = sum(np.array(output == test_y))

            accuracy.append((correct* 1.0)/ len(output))
            
            # calculate roc curve
            fpr, tpr, n_neighbors = roc_curve(output, test_y,drop_intermediate=False)
            
            auc = metrics.auc(fpr, tpr)#calculating auc for tpr and fpr
            
        # plot no skill
        pyplot.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        pyplot.plot(fpr, tpr, marker='.')
        txt="AUC",metrics.auc(fpr, tpr)
        plt.text(0.7,0.2,txt,ha='center')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        # show the plot
        pyplot.show()
        
        print("average accuracy KNN", (sum(accuracy)/ 10.0)*100)

KNN()


# In[ ]:




