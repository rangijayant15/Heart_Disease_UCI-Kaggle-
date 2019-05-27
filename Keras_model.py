#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv('heart.csv')
#print(data)


# In[3]:


# Features preprocessing using basic feature engeneering
# Feature engeneering is done using trends in the bar graphs for different features
def age(number):
    if(number<53):
        return 0
    elif(number<63):
        return 1
    elif(number>=63):
        return 2
    
data['age']=data['age'].apply(age)
def trestbps(val):
    if(val<125):
        return 0
    if(val<=147):
        return 1
    if(val>147):
        return 3
data['trestbps']=data['trestbps'].apply(trestbps) 
def chol(val):
    if(val<258):
        return 0
    else:
        return 1
data['chol'] =data['chol'].apply(chol)
def thalach(val):
    if val<170:
        return 0
    else:
        return 1
data['thalach']=data['thalach'].apply(thalach)    
def oldpeak(val):
    if val<=0.62 :
        return 0
    else:
        return 1
data['oldpeak']=data['oldpeak'].apply(oldpeak)    



# In[4]:


#print(data.columns)
X=data.iloc[:,0:13].values
y=data.iloc[:,13].values
#print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[5]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[38]:


classifier = Sequential()
# Adding the input layer 
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 13))
# Adding the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[39]:


# Fitting our model 
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 200)


# In[40]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# In[41]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test,y_pred))
print(cm)


# In[42]:


#Checking the accuracy on Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
pred=clf.predict(X_test)
print(accuracy_score(y_test,pred))


# In[ ]:




