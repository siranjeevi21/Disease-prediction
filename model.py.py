#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn


# In[2]:


df = pd.read_csv('dataset.csv')
df.head()


# In[3]:


cols = df.columns
data = df[cols].values.flatten()


# In[4]:


s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)


# In[5]:


df = pd.DataFrame(s, columns=df.columns)
df = df.fillna(0)


# In[6]:


df1 = pd.read_csv('Symptom-severity.csv')
vals = df.values


# In[7]:


symptoms = df1['Symptom'].unique()


# In[8]:


for i in range(len(symptoms)):
    vals[vals==symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]


# In[9]:


d = pd.DataFrame(vals, columns=cols)
d.head(100)


# In[10]:


import joblib
import streamlit


# In[11]:


d = d.replace('dischromic _patches', 0)
d = d.replace('spotting_ urination', 0)
df = d.replace('foul_smell_of urine', 0)
df.head()


# In[12]:


data = df.iloc[:, 1:].values
labels = df['Disease'].values


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, train_size=0.85)


# In[15]:


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[16]:


from sklearn.svm import SVC


# In[17]:


model = SVC()
model.fit(x_train, y_train)


# In[18]:


preds = model.predict(x_test)


# In[28]:


preds


# In[19]:


from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns


# In[ ]:





# In[20]:


conf_mat = confusion_matrix(y_test, preds)
df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)
sns.heatmap(df_cm)


# In[21]:


model.score(data, labels)


# In[22]:


joblib.dump(model, 'svc_model.pkl')


# In[23]:


df.head()


# In[25]:


df.head(50)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




