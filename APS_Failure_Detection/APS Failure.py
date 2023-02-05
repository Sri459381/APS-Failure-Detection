#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# 
# # Load Dataset

# In[2]:


train = pd.read_csv("./Documents/PROJECTS/APS Failure detection/aps_failure_training_set.csv")
test  = pd.read_csv("./Documents/PROJECTS/APS Failure detection/aps_failure_test_set.csv")


# In[3]:


print(train.shape)
print(test.shape)


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


data = pd.concat([train,test]).reset_index(drop=True)


# In[7]:


data.shape


# In[8]:


data.head()


# In[9]:


data['class'].unique()


# In[10]:


data['class'].value_counts()


# In[11]:


data.info()


# In[12]:


data.describe()


# In[13]:


data.describe(include='O')


# In[14]:


data.duplicated().sum()


# In[15]:


df = data.copy()


# In[16]:


df.shape


# In[17]:


df.info(max_cols=172, show_counts=True)


# In[18]:


num_f = [feature for feature in df.columns if df[feature].dtype != 'O']
cat_f = [feature for feature in df.columns if df[feature].dtype == 'O']

print("we have {} numerical features: {}".format(len(num_f), num_f))
print("we have {} categorical features: {}".format(len(cat_f), cat_f))


# In[19]:


for col in cat_f:
    print(df[col].value_counts(normalize=True)*100)
    print('---------------------------')


# In[20]:


df.replace(to_replace=['na','nan'], value=np.NaN, inplace=True)


# In[21]:


df.isna().sum()


# In[22]:


nul_values = pd.DataFrame(df.isnull().sum(), columns = ["No.of null values"])
nul_values.loc[:, '%of null values'] = np.round(nul_values.loc[:,'No.of null values']/ df.shape[0]*100, 2)
nul_values.loc[nul_values.loc[:,"No.of null values"]>0, :].sort_values(by = "No.of null values", ascending = True)
pd.set_option('display.max_rows', None)
nul_values


# In[23]:


df['class'] = df['class'].map({'neg':0, 'pos':1})


# In[24]:


df['class'].value_counts()


# In[25]:


sns.countplot(df['class'])


# In[26]:


drop_column = df.columns[df.isnull().mean()>0.3]
drop_column


# In[27]:


df.drop(drop_column, axis=1, inplace=True)


# In[28]:


df.shape


# In[29]:


num_fs = [feature for feature in df.columns if df[feature].dtype != 'O']
cat_fs = [feature for feature in df.columns if df[feature].dtype == 'O']

print("we have {} numerical features :{}".format(len(num_fs), num_fs))
print("we have {} categorical features :{}".format(len(cat_fs), cat_fs))


# In[30]:


for col in df:
    df[col].fillna(df[col].median(), inplace=True)


# In[31]:


df.isnull().sum()


# In[32]:


for i in cat_fs:
    if df[i].dtype == 'O':
        df[i] = df[i].astype('float')


# In[33]:


df.info()


# In[34]:


num_fs = [feature for feature in df.columns if df[feature].dtype != 'O']
cat_fs = [feature for feature in df.columns if df[feature].dtype == 'O']

print("we have {} numerical features :{}".format(len(num_fs), num_fs))
print('======================================================================')
print("we have {} categorical features :{}".format(len(cat_fs), cat_fs))


# In[35]:


for i in num_fs:
    if df[i].dtype != 'O':
        df[i]=df[i].astype('int')


# In[36]:


df.head()


# # Outliers Detection and Removal

# In[37]:


#IQR method

def outlier_thresholds(dataframe, variable):
    Q1 = dataframe[variable].quantile(0.25)
    Q3 = dataframe[variable].quantile(0.75)
    IQR = Q3 - Q1
    up_limit = Q3 + 1.5 * IQR
    low_limit = Q1 - 1.5 * IQR
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    return dataframe


# In[38]:


for feature in num_fs[1:]:
    df = outlier_thresholds(df, feature)


# In[39]:


df.shape


# In[ ]:


df


# # Split the data into X and y

# In[42]:


x = df.drop(['class'], axis=1)
y = df['class']


# In[43]:


x.head()


# In[44]:


y


# In[45]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 716785)


# In[46]:


x_train.shape, y_train.shape


# In[47]:


x_test.shape, y_test.shape


# In[48]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# Model Building

# Model 1 : Using Logistic Regression

# In[49]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression


# In[50]:


log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
y_pred1 = log_reg.predict(x_test)


# In[51]:


log_reg.score(x_train, y_train)


# In[52]:


acc_log = accuracy_score(y_test, y_pred1)
acc_log


# In[53]:


conf_mat_log = confusion_matrix(y_test, y_pred1)
conf_mat_log


# In[69]:





# In[54]:


import pickle


# In[55]:


filename = "APS_Failure.sav"
pickle.dump(log_reg, open(filename, 'wb'))


# In[56]:


#Loading the saved model

loaded_model = pickle.load(open('APS_Failure.sav', 'rb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




