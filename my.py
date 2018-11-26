
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn import metrics


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

def LR():
    #linear regreesion
    
    # threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    kf = KFold(n_splits=10)
    K = 10
    # for i in threshold:
    accuracy = []

    fpr = [0, 0,0]
    tpr = [0,0,0]


    for training_id, test_id in kf.split(total_X):
        train_x, test_x = total_X[training_id], total_X[test_id]
        train_y, test_y = total_y[training_id], total_y[test_id]

        train_y = train_y.astype('int') 
        clf = LinearRegression()
        clf.fit(train_x, train_y)

        output = clf.predict(test_x)
        # print(np.dot(test_x, clf.coef_))
        # correct = sum(np.array(np.dot(test_x, clf.coef_) > i) == test_y)

        # accuracy.append((correct* 1.0)/ len(output))

        predict = (2 * (clf.predict(test_x) > 0.5)) - 1
        # print("predict ", predict)

        for loc in range(len(predict)):
            if predict[loc] == -1:
                predict[loc] = 0

        # print("predict 2 ", predict)
        correct = sum(np.array(predict == test_y))

        f, t,_=roc_curve(predict ,test_y,drop_intermediate=False)
        # print(fpr, tpr)

        if len(f) >2:
            fpr[0] += f[0]
            fpr[1] += f[1]
            fpr[2] += f[2]

            tpr[0] += t[0]
            tpr[1] += t[1]
            tpr[2] += t[2]



        accuracy.append((correct* 1.0)/ len(output))

    fpr = fpr[0]/K, fpr[1]/K, fpr[2]/K
    tpr = tpr[0]/K, tpr[1]/K, tpr[2]/K

    fig = plt.figure()

    #Create ROC
    plt.plot(fpr, tpr, color='red',lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title("ROC curve")
    txt = "AUC ", metrics.auc(fpr, tpr)
    fig.text(.7, .2, txt, ha='center')
    plt.show()

    # print("fpr ---------> ",i,  fpr)
    # print("tpr ---------> ",i,  tpr)
    print("AUC ---------> ",   metrics.auc(fpr, tpr))

    print("average accuracy LR", sum(accuracy)/ 10.0)

LR()

# SVM_implementation()
# NaiveBayes()
# DecisionTree()
# LDA()
# RF()