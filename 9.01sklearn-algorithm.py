#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# > Through a series processing, compressed 6 million rows of data into 20 thousands rows.

# In[2]:


df = pd.read_csv('https://www.kaggle.com/ntnu-testimon/paysim1?select=PS_20174392719_1491204439457_log.csv')
df.head()


# In[3]:


df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
df.head()


# In[4]:


df.shape


# In[7]:


#Storing the fraudulent data into a dataframe
df_fraud = df[df['isFraud'] == 1]


# In[8]:


#Storing the non-fraudulent data into a dataframe 
df_nofraud = df[df['isFraud'] == 0]
#Storing 12,000 rows of non-fraudulent data
df_nofraud = df_nofraud.head(12000)
#Joining both datasets together 
df = pd.concat([df_fraud, df_nofraud], axis = 0)
df.shape


# In[9]:


#Package Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#Converting the type column to categorical
df['type'] = df['type'].astype('category')

#Integer Encoding the 'type' column
type_encode = LabelEncoder()

#Integer encoding the 'type' column
df['type'] = type_encode.fit_transform(df.type)


# In[10]:


df.head(120)


# In[11]:


#One hot encoding the 'type' column
type_one_hot = OneHotEncoder()
type_one_hot_encode = type_one_hot.fit_transform(df.type.values.reshape(-1,1)).toarray()

#Adding the one hot encoded variables to the dataset 
ohe_variable = pd.DataFrame(type_one_hot_encode, columns = ["type_"+str(int(i)) for i in range(type_one_hot_encode.shape[1])])
df = pd.concat([df, ohe_variable], axis=1)

#Dropping the original type variable 
df = df.drop('type', axis = 1)

#Viewing the new dataframe after one-hot-encoding 
df.head()


# In[12]:


df.isnull().any()


# In[2]:


import pandas as pd
df = pd.read_csv('fraud_prediction.csv')


# In[3]:


df = df.fillna(0)
df


# In[4]:


df.to_csv('fraud_prediction.csv')


# In[ ]:




