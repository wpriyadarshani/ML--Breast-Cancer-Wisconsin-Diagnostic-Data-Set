#!/usr/bin/env python
# coding: utf-8

# In[98]:


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


# In[99]:


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


# In[128]:


def LogisticReg():
    
    kf = KFold(n_splits=10)
        # predict probabilities
    thresholds=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for thres in thresholds:
        accuracy = []
        TPR_list = [0,0,0]
        FPR_list = [0,0,0]
        for training_id, test_id in kf.split(total_X):
            train_x, test_x = total_X[training_id], total_X[test_id]
            train_y, test_y = total_y[training_id], total_y[test_id]

            train_y = train_y.astype('int') 
            log_r = LogisticRegression(solver='lbfgs')
            log_r.fit(train_x, train_y)

            output = log_r.predict(test_x)

            preds=np.where(log_r.predict_proba(test_x)[:,1]>thres,1,0)

            fpr, tpr, _ = roc_curve(preds,test_y, drop_intermediate=False)
            
            print('Accuracy from scratch: {0}'.format((preds == test_y).sum().astype(float) / len(preds)))
            
            accuracy.append((preds == test_y).sum().astype(float) / len(preds))
            
            if len(tpr) > 2:
                TPR_list[0]+=tpr[0]
                TPR_list[1]+=tpr[1]
                TPR_list[2]+=tpr[2]

            if len(fpr) > 2:
                FPR_list[0]+=fpr[0]
                FPR_list[1]+=fpr[1]
                FPR_list[2]+=fpr[2]
        

        #Calculating the average of TPR and FPR
        FPR_Avg = FPR_list[0]/10.0, FPR_list[1]/10.0,FPR_list[2]/10.0 
        TPR_Avg = TPR_list[0]/10.0, TPR_list[1]/10.0,TPR_list[2]/10.0

        auc = metrics.auc(FPR_Avg, TPR_Avg)#calculating auc for tpr and fpr
        print('fpr',FPR_Avg)
        print('tpr',TPR_Avg)
        print('auc',auc)

        #plotting graph for ROC
        plt.figure()
        lw = 2
        plt.plot(FPR_Avg, TPR_Avg, color='darkorange',lw=lw)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        txt="AUC",metrics.auc(FPR_Avg, TPR_Avg)
        plt.text(0.7,0.2,txt,ha='center')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        
        print("average=", sum(accuracy)/ 10.0)

LogisticReg()


# In[ ]:




