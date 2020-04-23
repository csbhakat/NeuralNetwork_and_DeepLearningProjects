# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:06:38 2020

@author: chbhakat
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
from keras.layers import Dense 
from keras.layers import LeakyReLU,PReLU, ELU
from keras.layers import Dropout

#Initializing the ANN
classifier=Sequential()

#adding input layer and first hidden layer
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=14))

#adding next hiddel layers with 6 neurons
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))

#adding output layer
classifier.add(Dense(output_dim=1,init='glorot_uniform',activation='sigmoid'))

classifier.compile(optimizer='Adamax',loss='binary_crossentropy',metrics=['accuracy'])

#train the model

classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,nb_epoch=100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)

