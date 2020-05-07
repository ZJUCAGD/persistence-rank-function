import glob 
import numpy as np
import warnings
import sys
warnings.filterwarnings('ignore')

## training data loading
tr_path = str(sys.argv[1]) + '/*.txt'
tr_files = glob.glob(tr_path)
tr_files = sorted(tr_files)

# data list
data = []
for file in tr_files:
    data.append(np.loadtxt(file))
data = np.array(data)


## labels
label_path = str(sys.argv[3])
label_file = glob.glob(label_path)
label = []
for file in label_file:
    label.append(np.loadtxt(file))
label = np.array(label)
label = np.transpose(label)
   
train = np.c_[data,label]
print('Shape of training data+labels:')
print(train.shape)

## test data loading
ts_path = str(sys.argv[2]) + '/*.txt'
ts_files = glob.glob(ts_path)
ts_files = sorted(ts_files)

# data list
data = []
for file in ts_files:
    data.append(np.loadtxt(file))
data = np.array(data)

# labels
label_path = str(sys.argv[4])
label_file = glob.glob(label_path)
label = []
for file in label_file:
    label.append(np.loadtxt(file))
label = np.array(label)
label = np.transpose(label)

test = np.c_[data,label]
print('Shape of test data+labels:')
print(test.shape)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# %matplotlib inline

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC



classifiers = [
    KNeighborsClassifier(3),
    LogisticRegression(),
    LinearSVC(C=1),
    SVC(kernel = 'rbf', C = 1, gamma = 0.01)
]

log_cols = ['classifiers', 'accuracy']
log = pd.DataFrame(columns=log_cols)

train_data = train[:,:-1]
train_labels = train[:,-1]

test_data = test[:,:-1]
test_labels = test[:,-1]

acc_dict = {}

clf1 = LogisticRegression()
name= clf1.__class__.__name__
clf1.fit(train_data, train_labels)
train_predictions = clf1.predict(test_data)
acc = accuracy_score(test_labels, train_predictions)
acc_dict[name] = acc

ks = [2,3,4,5,6,7,8,10,15,20]
acc_max = 0
k_opt = 0

for k in ks:
    clf2 = KNeighborsClassifier(k)
    clf2.fit(train_data, train_labels)
    train_predictions = clf2.predict(test_data)
    acc = accuracy_score(test_labels, train_predictions)
    if acc>acc_max:
        acc_max = acc
        k_opt = k
name= clf2.__class__.__name__
acc_dict[name] = acc_max

Cs=[0.001, 0.01, 0.1, 1, 5, 10, 100, 1000]
gammas = [0.001, 0.01,0.05, 0.1,0.5, 1, 5, 10]

acc_max = 0
C_opt1 = 0
gamma_opt = 0

for C in Cs:
    for gamma in gammas:
        log  = pd.DataFrame(columns=log_cols)
        clf3 = SVC(C=C, gamma=gamma)
        clf3.fit(train_data, train_labels)
        train_predictions = clf3.predict(test_data)
        acc = accuracy_score(test_labels, train_predictions)
        if acc>acc_max:
            acc_max = acc
            C_opt1 = C
            gamma_opt = gamma

name = clf3.__class__.__name__
acc_dict[name] = acc_max


acc_max = 0
C_opt2 = 0

for C in Cs:
    for gamma in gammas:
        log  = pd.DataFrame(columns=log_cols)
        clf4 = LinearSVC(C=C)
        clf4.fit(train_data, train_labels)
        train_predictions = clf4.predict(test_data)
        acc = accuracy_score(test_labels, train_predictions)
        if acc>acc_max:
            acc_max = acc
            C_opt2 = C

name = clf4.__class__.__name__
acc_dict[name] = acc_max
print("Accuracy of classification")

for clf in acc_dict:
    log_entry = pd.DataFrame([[clf,acc_dict[clf]]],columns = log_cols)
    log = log.append(log_entry)
    print(clf)
    print(acc_dict[clf])
    print(" ")
log

print("Hyperparameter Selection:")
print("KNN: k_opt =" + str(k) + "\n")
print("KernelSVM: C = " + str(C_opt1)+",  gamma = " + str(gamma_opt)+"\n")
print("LinearSVM: C = " + str(C_opt2)+"\n")
