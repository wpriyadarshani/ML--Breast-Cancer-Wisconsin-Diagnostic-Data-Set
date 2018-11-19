
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier


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

#SVM 

#cross validate data
def SVM_implementation():
    accuracy = []
    kf = KFold(n_splits=10)
    for training_id, test_id in kf.split(total_X):
        train_x, test_x = total_X[training_id], total_X[test_id]
        train_y, test_y = total_y[training_id], total_y[test_id]

        train_y = train_y.astype('int') 
        clf = svm.SVC(gamma='scale')
        clf.fit(train_x, train_y)

        output = clf.predict(test_x)

        # print(output)

        correct = sum(np.array(output == test_y))

        accuracy.append((correct* 1.0)/ len(output))


    print("average accuracy SVM", sum(accuracy)/ 10.0)



#cross validate data
def NaiveBayes():
    accuracy = []
    kf = KFold(n_splits=10)
    for training_id, test_id in kf.split(total_X):
        train_x, test_x = total_X[training_id], total_X[test_id]
        train_y, test_y = total_y[training_id], total_y[test_id]

        train_y = train_y.astype('int') 
        clf = GaussianNB()
        clf.fit(train_x, train_y)

        output = clf.predict(test_x)
        correct = sum(np.array(output == test_y))

        # print(correct)

        accuracy.append((correct* 1.0)/ len(output))


    print("average accuracy NaiveBayes", sum(accuracy)/ 10.0)

def DecisionTree():
    accuracy = []
    kf = KFold(n_splits=10)
    for training_id, test_id in kf.split(total_X):
        train_x, test_x = total_X[training_id], total_X[test_id]
        train_y, test_y = total_y[training_id], total_y[test_id]

        train_y = train_y.astype('int') 
        clf = tree.DecisionTreeClassifier()
        clf.fit(train_x, train_y)

        output = clf.predict(test_x)
        correct = sum(np.array(output == test_y))

        # print(correct)

        accuracy.append((correct* 1.0)/ len(output))


    print("average accuracy DecisionTree", sum(accuracy)/ 10.0)

def LDA():
    accuracy = []
    kf = KFold(n_splits=10)
    for training_id, test_id in kf.split(total_X):
        train_x, test_x = total_X[training_id], total_X[test_id]
        train_y, test_y = total_y[training_id], total_y[test_id]

        train_y = train_y.astype('int') 
        clf = LinearDiscriminantAnalysis()
        clf.fit(train_x, train_y)

        output = clf.predict(test_x)
        correct = sum(np.array(output == test_y))

        # print(correct)

        accuracy.append((correct* 1.0)/ len(output))


    print("average accuracy LDA", sum(accuracy)/ 10.0)

def RF():
    accuracy = []
    kf = KFold(n_splits=10)
    for training_id, test_id in kf.split(total_X):
        train_x, test_x = total_X[training_id], total_X[test_id]
        train_y, test_y = total_y[training_id], total_y[test_id]

        train_y = train_y.astype('int') 
        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        clf.fit(train_x, train_y)

        output = clf.predict(test_x)
        correct = sum(np.array(output == test_y))

        # print(correct)

        accuracy.append((correct* 1.0)/ len(output))


    print("average accuracy Random Forest", sum(accuracy)/ 10.0)


# SVM_implementation()
# NaiveBayes()
# DecisionTree()
# LDA()
RF()