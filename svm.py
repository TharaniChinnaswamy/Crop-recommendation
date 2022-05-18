# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:41:17 2022

@author: Tharani
"""
# Load libraries
import pickle 
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score
from sklearn import svm




#col_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall','label']
tree = pd.read_csv('Crop_recommendation.csv', encoding='utf-8')
tree.head()
#split dataset in features and target variable
#feature_cols = ['N', 'P', 'K', 'temperature','humidity','ph','rainfall']
#X = tree[feature_cols] # Features
#y = tree.label # Target variable


X = tree.iloc[:,0:7].values # Features
y = tree.iloc[:,7].values # Target variable


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


 

clf1 = svm.SVC(kernel='linear')
clf1.fit(X_train,y_train)
#pred=clf.predict
y_pred1 = clf1.predict(X_test)
print("SVM Accurancy:",(metrics.accuracy_score(y_test, y_pred1)*100))

pickle.dump(clf1,open('model.pkl','wb'))
