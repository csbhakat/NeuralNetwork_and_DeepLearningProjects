# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:24:31 2020

@author: chbhakat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Users/chbhakat/Desktop/DS/Kaggle/DeepLearningProjects/Churn_Modelling.csv")



df=pd.get_dummies(df,columns=['Geography','Gender'])

X=df.iloc[:,3:17]
y=df.iloc[:,11]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Embedding,LeakyReLU,PReLU, ELU,BatchNormalization
from keras.layers import Dropout
from keras.activations import relu, sigmoid
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def create_model(activation,layers):
    classifier=Sequential()
    for i, nodes in enumerate(layers):
        if(i==0):
            classifier.add(Dense(units=layers,kernel_initializer='he_uniform',activation=activation,input_dim=X_train.shape[1]))
        else:
            classifier.add(Dense(units=layers,kernel_initializer='he_uniform',activation=activation))
            
    classifier.add(Dense(output_dim=1,init='glorot_uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=create_model, verbose=0)
layers = [[20], [40, 20], [45, 30, 15]]
activations = ['sigmoid', 'relu'] 

param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30])
grid = GridSearchCV(estimator=classifier, param_grid=param_grid,cv=5)

grid_result = grid.fit(X_train, y_train)
print(grid_result.best_score_,grid_result.best_params_)
           
        
    
    