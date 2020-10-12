# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:40:13 2020

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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt



DF =pd.read_csv('CA_MST_ML.csv')
DF['Label']=np.where(DF['Human']>DF['Nonhuman'],1,0)
Data=DF.loc[:,['L11K2', 'L2K2', 'L31K2', 'L4K2', 'L52K2', 'L71K2', 'L8K2', 'L9K2','Prep0_mm', 'Temp0_c','Elv', 'Flowacc','Label']]


data =Data.values #convert data from dataframe to numpy array

xdata = data[:,0:12]
ydata = data[:,12]

#Split the data into training and test data by 75:25
x_train, x_test, y_train, y_test =model_selection.train_test_split(xdata, ydata, test_size =0.2, random_state=48)
xtrain_scaled = preprocessing.scale(x_train)
scaler = preprocessing.StandardScaler().fit(x_train)
xtest_scaled =scaler.transform(x_test)

## KNN
KNN_model = neighbors.KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(xtrain_scaled, y_train)
KNN_pred = KNN_model.predict(xtest_scaled)
KNN_accuracy= sum(KNN_pred==y_test)/len(y_test)
KNN_predp = KNN_model.predict_proba(xtest_scaled)
KNN_predp=KNN_predp[:, 1]
KNN_AUC1=roc_auc_score(y_test,KNN_predp)
KNN_fpr1, KNN_tpr1, _ = roc_curve(y_test,KNN_predp)
print('KNN: ', 'Accuracy=', KNN_accuracy, 'AUC=', KNN_AUC1)


## Naive Bayes
NB_model = naive_bayes.GaussianNB()
NB_model.fit(xtrain_scaled, y_train)
NB_pred = NB_model.predict(xtest_scaled)
NB_accuracy= sum(NB_pred==y_test)/len(y_test)
NB_predp = NB_model.predict_proba(xtest_scaled)
NB_predp=NB_predp[:, 1]
NB_AUC1=roc_auc_score(y_test,NB_predp)
print('Naive Bayes: ', 'Accuracy=', NB_accuracy, 'AUC=', NB_AUC1)
NB_fpr1, NB_tpr1, _ = roc_curve(y_test,NB_predp)


#  SVC method
SVC_model= SVC(gamma='auto', C=1, max_iter=2000,probability=True,random_state=90, tol=1e-4)
SVC_model.fit(xtrain_scaled, y_train)
SVC_pred =SVC_model.predict(xtest_scaled)
SVC_accuracy= sum(SVC_pred==y_test)/len(y_test)
SVC_predp = SVC_model.predict_proba(xtest_scaled)
SVC_predp=SVC_predp[:, 1]
SVC_AUC1=roc_auc_score(y_test,SVC_predp)
SVC_fpr1, SVC_tpr1, _ = roc_curve(y_test,SVC_predp)
print('Support Vector Machine: ', 'Accuracy=', SVC_accuracy, 'AUC=', SVC_AUC1)

## Neural network
NN_model = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(5, 3),random_state=48)
NN_model.fit(xtrain_scaled, y_train)
NN_pred = NN_model.predict(xtest_scaled)
NN_accuracy= sum(NN_pred==y_test)/len(y_test)
NN_predp = NN_model.predict_proba(xtest_scaled)
NN_predp=NN_predp[:, 1]
NN_AUC1=roc_auc_score(y_test,NN_predp)
NN_fpr1, NN_tpr1, _ = roc_curve(y_test,NN_predp)
print('Neural Network: ', 'Accuracy=', NN_accuracy, 'AUC=', NN_AUC1)


## Random Forest
RF_model =RandomForestClassifier(n_estimators = 15, random_state =90)
RF_model.fit(xtrain_scaled, y_train)
RF_pred =RF_model.predict(xtest_scaled)
RF_accuracy=sum(RF_pred==y_test)/len(y_test)
RF_predp = RF_model.predict_proba(xtest_scaled)
RF_predp=RF_predp[:, 1]
RF_AUC1=roc_auc_score(y_test,RF_predp)
RF_fpr1, RF_tpr1, _ = roc_curve(y_test,RF_predp)
print('Random Forest: ', 'Accuracy=', RF_accuracy, 'AUC=', RF_AUC1)
RF_model.feature_importances_


## XGBoost
XGB_model =XGBClassifier(n_estimators=30, random_state=90)
XGB_model.fit(xtrain_scaled, y_train)
XGB_pred =XGB_model.predict(xtest_scaled)
XGB_accuracy=sum(XGB_pred==y_test)/len(y_test)
XGB_predp = XGB_model.predict_proba(xtest_scaled)
XGB_predp=XGB_predp[:, 1]
XGB_AUC1=roc_auc_score(y_test,XGB_predp)
XGB_fpr1, XGB_tpr1, _ = roc_curve(y_test,XGB_predp)
print('XGBoost: ', 'Accuracy=', XGB_accuracy, 'AUC=', XGB_AUC1)



#######################################################
## Hydrological feature #########################
#######################################################

Data=DF.loc[:,['HYL11', 'HYL2', 'HYL31', 'HYL4', 'HYL52', 'HYL71', 'HYL8', 'HYL9', 'Prep0_mm', 'Temp0_c','Elv', 'Flowacc','Label']]

data =Data.values #convert data from dataframe to numpy array

xdata = data[:,0:12]
ydata = data[:,12]

#Split the data into training and test data by 75:25
x_train, x_test, y_train, y_test =model_selection.train_test_split(xdata, ydata, test_size =0.2, random_state=48)
xtrain_scaled = preprocessing.scale(x_train)
scaler = preprocessing.StandardScaler().fit(x_train)
xtest_scaled =scaler.transform(x_test)

## KNN
KNN_model = neighbors.KNeighborsClassifier(n_neighbors=3)
KNN_model.fit(xtrain_scaled, y_train)
KNN_pred = KNN_model.predict(xtest_scaled)
KNN_accuracy= sum(KNN_pred==y_test)/len(y_test)
KNN_predp = KNN_model.predict_proba(xtest_scaled)
KNN_predp=KNN_predp[:, 1]
KNN_AUC2=roc_auc_score(y_test,KNN_predp)
KNN_fpr2, KNN_tpr2, _ = roc_curve(y_test,KNN_predp)
print('KNN: ', 'Accuracy=', KNN_accuracy, 'AUC=',KNN_AUC2 )


## Naive Bayes
NB_model = naive_bayes.GaussianNB(priors=None, var_smoothing=1e-06)
NB_model.fit(xtrain_scaled, y_train)
NB_pred = NB_model.predict(xtest_scaled)
NB_accuracy= sum(NB_pred==y_test)/len(y_test)
NB_predp = NB_model.predict_proba(xtest_scaled)
NB_predp=NB_predp[:, 1]
NB_AUC2=roc_auc_score(y_test,NB_predp)
NB_fpr2, NB_tpr2, _ = roc_curve(y_test,NB_predp)
print('Naive Bayes: ', 'Accuracy=', NB_accuracy, 'AUC=', NB_AUC2)


# linear SVC method
SVC_model= SVC(gamma='auto', C=1.0, max_iter=2000,probability=True,random_state=10, tol=1e-4)
SVC_model.fit(xtrain_scaled, y_train)
SVC_pred =SVC_model.predict(xtest_scaled)
SVC_accuracy= sum(SVC_pred==y_test)/len(y_test)
SVC_predp = SVC_model.predict_proba(xtest_scaled)
SVC_predp=SVC_predp[:, 1]
SVC_AUC2=roc_auc_score(y_test,SVC_predp)
SVC_fpr2, SVC_tpr2, _ = roc_curve(y_test,SVC_predp)
print('Support Vector Machine: ', 'Accuracy=', SVC_accuracy, 'AUC=', SVC_AUC2)


## Neural network
NN_model = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(5, 2),random_state=48)
NN_model.fit(xtrain_scaled, y_train)
NN_pred = NN_model.predict(xtest_scaled)
NN_accuracy= sum(NN_pred==y_test)/len(y_test)
NN_predp = NN_model.predict_proba(xtest_scaled)
NN_predp=NN_predp[:, 1]
NN_AUC2=roc_auc_score(y_test,NN_predp)
NN_fpr2, NN_tpr2, _ = roc_curve(y_test,NN_predp)
print('Neural network: ', 'Accuracy=', NN_accuracy, 'AUC=', NN_AUC2)


## Random Forest
RF_model =RandomForestClassifier(n_estimators = 20,  random_state =41)
RF_model.fit(xtrain_scaled, y_train)
RF_pred =RF_model.predict(xtest_scaled)
RF_accuracy=sum(RF_pred==y_test)/len(y_test)
RF_predp = RF_model.predict_proba(xtest_scaled)
RF_predp=RF_predp[:, 1]
RF_AUC2=roc_auc_score(y_test,RF_predp)
RF_fpr2, RF_tpr2, _ = roc_curve(y_test,RF_predp)
print('Random Forest: ', 'Accuracy=', RF_accuracy, 'AUC=', RF_AUC2)


# importance of features
IF=RF_model.feature_importances_


## XGBoost
XGB_model =XGBClassifier(n_estimators=10, random_state=48)
XGB_model.fit(xtrain_scaled, y_train)
XGB_pred =XGB_model.predict(xtest_scaled)
XGB_accuracy=sum(XGB_pred==y_test)/len(y_test)

XGB_predp = XGB_model.predict_proba(xtest_scaled)
XGB_predp=XGB_predp[:, 1]
XGB_AUC2=roc_auc_score(y_test,XGB_predp)
XGB_fpr2, XGB_tpr2, _ = roc_curve(y_test,XGB_predp)
print('XGBoost: ', 'Accuracy=', XGB_accuracy, 'AUC=', XGB_AUC2)






##################### Subplot templete  #############
# calculate roc curves

fig, axs = plt.subplots(3, 2)


axs[0, 0].plot(KNN_fpr1, KNN_tpr1, marker='.', label = 'Group 1 (AUC = %0.2f)' % KNN_AUC1)
axs[0, 0].plot(KNN_fpr2, KNN_tpr2, marker='.', label = 'Group 2 (AUC = %0.2f)' % KNN_AUC2)
axs[0, 0].set_title('K-Nearest Neighbors') 
axs[0, 0].set(ylabel='True Positive Rate') 
axs[0, 0].legend(loc = 'lower right')

axs[0, 1].plot(NB_fpr1, NB_tpr1, marker='.', label = 'Group 1 (AUC = %0.2f)' % NB_AUC1)
axs[0, 1].plot(NB_fpr2, NB_tpr2, marker='.', label = 'Group 2 (AUC = %0.2f)' % NB_AUC2)
axs[0, 1].set_title('Naive Bayes') 
axs[0, 1].legend(loc = 'lower right')



axs[1, 0].plot(SVC_fpr1, SVC_tpr1, marker='.', label = 'Group 1 (AUC = %0.2f)' % SVC_AUC1)
axs[1, 0].plot(SVC_fpr2, SVC_tpr2, marker='.', label = 'Group 2 (AUC = %0.2f)' % SVC_AUC2)
axs[1, 0].set_title('Support Vector Machine') 
axs[1, 0].set(ylabel='True Positive Rate') 
axs[1, 0].legend(loc = 'lower right')

axs[1, 1].plot(NN_fpr1, NN_tpr1, marker='.', label = 'Group 1 (AUC = %0.2f)' % NN_AUC1)
axs[1, 1].plot(NN_fpr2, NN_tpr2, marker='.', label = 'Group 2 (AUC = %0.2f)' % NN_AUC2)
axs[1, 1].set_title('Neural Network') 
axs[1, 1].legend(loc = 'lower right')



axs[2, 0].plot(RF_fpr1, RF_tpr1, marker='.', label = 'Group 1 (AUC = %0.2f)' % RF_AUC1)
axs[2, 0].plot(RF_fpr2, RF_tpr2, marker='.', label = 'Group 2 (AUC = %0.2f)' % RF_AUC2)
axs[2, 0].set_title('Random Forest') 
axs[2, 0].set(xlabel='False Positive Rate') 
axs[2, 0].set(ylabel='True Positive Rate') 
axs[2, 0].legend(loc = 'lower right')


axs[2, 1].plot(XGB_fpr1, XGB_tpr1, marker='.', label = 'Group 1 (AUC = %0.2f)' % XGB_AUC1)
axs[2, 1].plot(XGB_fpr2, XGB_tpr2, marker='.', label = 'Group 2 (AUC = %0.2f)' % XGB_AUC2)

axs[2, 1].set_title('XGBoost') 
axs[2, 1].set(xlabel='False Positive Rate') 
axs[2, 1].legend(loc = 'lower right')



### Important factor plot
x = ['% Water', '% Developed', '% Barren land', '% Forest', '% Shrubland', '% Grassland', '% Agriculture','% Wetland', 'Precipitation','Temperature',  'Elevation', 'Flow accumulation' ]
IFS = list(IF)


x_pos = [i for i, _ in enumerate(x)]

plt.barh(x_pos, IFS, color='cyan')
#plt.ylabel("Features (Predictor)")
plt.xlabel("Importance index")
plt.title('Importance of features (Predictors)')

plt.yticks(x_pos, x)
plt.grid()
plt.show()