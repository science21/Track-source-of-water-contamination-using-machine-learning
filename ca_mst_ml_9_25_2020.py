# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:22:46 2020

@author: Jianyong
"""

import os
path="D:\Data_analysis\Python\CA_MST"
os.chdir(path)

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn import preprocessing, model_selection
from sklearn import  naive_bayes,  neighbors
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score



DF =pd.read_csv('CA_MST_ML.csv')
DF['Label']=np.where(DF['Human']>DF['Nonhuman'],1,0)
Data=DF.loc[:,['L11K2', 'L2K2', 'L31K2', 'L4K2', 'L52K2', 'L71K2', 'L8K2', 'L9K2','Prep0_mm', 'Temp0_c','Elv', 'Flowacc','Label']]


data =Data.values #convert data from dataframe to numpy array

xdata = data[:,0:12]
ydata = data[:,12]

#Split the data into training and test data by 75:25
x_train, x_test, y_train, y_test =model_selection.train_test_split(xdata, ydata, test_size =0.20, random_state=48)
xtrain_scaled = preprocessing.scale(x_train)
scaler = preprocessing.StandardScaler().fit(x_train)
xtest_scaled =scaler.transform(x_test)

## KNN
KNN_model = neighbors.KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(xtrain_scaled, y_train)
KNN_pred = KNN_model.predict(xtest_scaled)
KNN_accuracy= sum(KNN_pred==y_test)/len(y_test)
KNN_AUC=roc_auc_score(y_test,KNN_pred)
print('KNN: ', 'Accuracy=', KNN_accuracy, 'AUC=',KNN_AUC)


## Naive Bayes
NB_model = naive_bayes.GaussianNB()
NB_model.fit(xtrain_scaled, y_train)
NB_pred = NB_model.predict(xtest_scaled)
NB_accuracy= sum(NB_pred==y_test)/len(y_test)
NB_AUC=roc_auc_score(y_test,NB_pred)
print('Naive Bayes: ', 'Accuracy=', NB_accuracy, 'AUC=',NB_AUC)

# linear SVC method
SVC_model= LinearSVC(C=1.0, max_iter=2000,random_state=90, tol=1e-4)
SVC_model.fit(xtrain_scaled, y_train)
SVC_pred =SVC_model.predict(xtest_scaled)
SVC_accuracy= sum(SVC_pred==y_test)/len(y_test)
SVC_AUC=roc_auc_score(y_test,SVC_pred)
print('Support Vector Machine: ', 'Accuracy=', SVC_accuracy, 'AUC=',SVC_AUC)



## Neural network
NN_model = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(5, 3),random_state=48)
NN_model.fit(xtrain_scaled, y_train)
NN_pred = NN_model.predict(xtest_scaled)
NN_accuracy= sum(NN_pred==y_test)/len(y_test)
NN_AUC=roc_auc_score(y_test,NN_pred)
print('Neural Network: ', 'Accuracy=', NN_accuracy, 'AUC=',NN_AUC)


## Random Forest
RF_model =RandomForestClassifier(n_estimators = 15, random_state =90)
RF_model.fit(xtrain_scaled, y_train)
RF_pred =RF_model.predict(xtest_scaled)
RF_accuracy=sum(RF_pred==y_test)/len(y_test)
RF_AUC=roc_auc_score(y_test,RF_pred)
print('Random Forest: ', 'Accuracy=', RF_accuracy, 'AUC=',RF_AUC)



## XGBoost
XGB_model =XGBClassifier(n_estimators=30, random_state=90)
XGB_model.fit(xtrain_scaled, y_train)
XGB_pred =XGB_model.predict(xtest_scaled)
XGB_accuracy=sum(XGB_pred==y_test)/len(y_test)
XGB_AUC=roc_auc_score(y_test,XGB_pred)
print('XGBoost: ', 'Accuracy=', XGB_accuracy, 'AUC=', XGB_AUC)





#######################################################
## Hydrological feature #########################
#######################################################

Data=DF.loc[:,['HYL11', 'HYL2', 'HYL31', 'HYL4', 'HYL52', 'HYL71', 'HYL8', 'HYL9', 'Prep0_mm', 'Temp0_c','Elv', 'Flowacc','Label']]

data =Data.values #convert data from dataframe to numpy array

xdata = data[:,0:12]
ydata = data[:,12]

#Split the data into training and test data by 75:25
x_train, x_test, y_train, y_test =model_selection.train_test_split(xdata, ydata, test_size =0.20, random_state=48)
xtrain_scaled = preprocessing.scale(x_train)
scaler = preprocessing.StandardScaler().fit(x_train)
xtest_scaled =scaler.transform(x_test)

## KNN
KNN_model = neighbors.KNeighborsClassifier(n_neighbors=3)
KNN_model.fit(xtrain_scaled, y_train)
KNN_pred = KNN_model.predict(xtest_scaled)
KNN_accuracy= sum(KNN_pred==y_test)/len(y_test)
KNN_AUC=roc_auc_score(y_test,KNN_pred)
print('KNN: ', 'Accuracy=', KNN_accuracy, 'AUC=',KNN_AUC)


## Naive Bayes
NB_model = naive_bayes.GaussianNB()
NB_model.fit(xtrain_scaled, y_train)
NB_pred = NB_model.predict(xtest_scaled)
NB_accuracy= sum(NB_pred==y_test)/len(y_test)
NB_AUC=roc_auc_score(y_test,NB_pred)
print('Naive Bayes: ', 'Accuracy=', NB_accuracy, 'AUC=',NB_AUC)


# linear SVC method
SVC_model= LinearSVC(C=1, max_iter=2000,random_state=48, tol=1e-4)
SVC_model.fit(xtrain_scaled, y_train)
SVC_pred =SVC_model.predict(xtest_scaled)
SVC_accuracy= sum(SVC_pred==y_test)/len(y_test)
SVC_AUC=roc_auc_score(y_test,SVC_pred)
print('Support Vector Machine: ', 'Accuracy=', SVC_accuracy, 'AUC=',SVC_AUC)


## Neural network
NN_model = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(5, 2),random_state=48)
NN_model.fit(xtrain_scaled, y_train)
NN_pred = NN_model.predict(xtest_scaled)
NN_accuracy= sum(NN_pred==y_test)/len(y_test)
NN_AUC=roc_auc_score(y_test,NN_pred)
print('Neural Network: ', 'Accuracy=', NN_accuracy, 'AUC=',NN_AUC)


## Random Forest
RF_model =RandomForestClassifier(n_estimators = 20,  random_state =41)
RF_model.fit(xtrain_scaled, y_train)
RF_pred =RF_model.predict(xtest_scaled)
RF_accuracy=sum(RF_pred==y_test)/len(y_test)
RF_AUC=roc_auc_score(y_test,RF_pred)
print('Random Forest: ', 'Accuracy=', RF_accuracy, 'AUC=',RF_AUC)

# importance of features
RF_model.feature_importances_


## XGBoost
XGB_model =XGBClassifier(n_estimators=10, random_state=48)
XGB_model.fit(xtrain_scaled, y_train)
XGB_pred =XGB_model.predict(xtest_scaled)
XGB_accuracy=sum(XGB_pred==y_test)/len(y_test)
XGB_AUC=roc_auc_score(y_test,XGB_pred)
print('XGBoost: ', 'Accuracy=', XGB_accuracy, 'AUC=', XGB_AUC)




