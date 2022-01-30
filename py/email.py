#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/gdrive')


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


# path = "/content/gdrive/MyDrive/dataset/"
path="../dataset new/"


# In[4]:


df_email = pd.read_csv(path+'email.csv')
display(df_email.shape)
display(df_email.head()) 


# In[5]:


df_user_email = pd.read_csv(path+'user_email.csv')
del df_user_email['Unnamed: 0']
display(df_user_email.shape)
display(df_user_email)


# In[6]:


df_email['from'].unique()


# In[7]:


display(len(df_email['from'].unique()))


# In[8]:


display(len(df_user_email['employee_name']))
display(len(df_user_email['email']))


# In[9]:


display(len(df_user_email['email'].unique()))


# In[10]:


df_user_email = df_user_email.drop_duplicates()


# In[11]:


email_copy = df_email.copy()


# In[12]:


user_email_copy = df_user_email.copy()


# In[13]:


email_user_merge = pd.merge(email_copy,user_email_copy,how="outer",indicator=True,left_on='from',right_on='email')
email_user_merge


# In[14]:


# email_user_merge['_merge'].unique()
# email_user_merge.isna().sum()


# In[15]:


# del email_user_merge['email']
# del email_user_merge['_merge']
del_columns = ['email','_merge','attachments','content','from']
email_user_merge.drop(columns=del_columns,inplace=True)

email_user_merge


# In[16]:


columns_reindex = ['id', 'date', 'employee_name','user_id','to','size']
email_user_merge = email_user_merge[columns_reindex]
email_user_merge


# In[17]:


email_user_merge['date'] = pd.to_datetime(email_user_merge['date'])


# In[18]:


email_user_merge['time'] = email_user_merge['date'].dt.time
email_user_merge['hour'] = email_user_merge['date'].dt.hour


# In[19]:


df_email_user = email_user_merge.copy()
df_email_user


# 

# In[20]:


df_email_user_stats = df_email_user.groupby('user_id')['time'].agg([min,max]).reset_index()
df_email_user_mode = df_email_user.groupby('user_id')['time'].agg(lambda x: x.value_counts().index[0]).reset_index()


# In[21]:


df_email_user_mean = df_email_user.groupby('user_id')['hour'].mean().reset_index()
df_email_user_mean['hour'] = df_email_user_mean['hour'].astype(int)
df_email_user_mean['hour'] = pd.to_datetime(df_email_user_mean['hour'],format="%H").dt.time


# In[22]:


df_email_user_mean


# 

# In[23]:


df_email_user_stats


# In[24]:


df_size_per_email = df_email_user.groupby(['user_id'])['size'].agg(sum).reset_index()


# In[25]:


df_size_per_email


# In[26]:


df_email_user_stats['mode'] = df_email_user_mode['time']
df_email_user_stats['mean'] = df_email_user_mean['hour']
df_email_user_stats['size'] = df_size_per_email['size']


# In[27]:


df_email_user_stats


# In[28]:


df_email_user_stats_sec = df_email_user_stats


# In[29]:


def dtt2timestamp(dtt):
  time_in_sec = (dtt.hour*60 + dtt.minute) * 60 + dtt.second
  return time_in_sec


# In[30]:


min_ts = [dtt2timestamp(dtt) for dtt in df_email_user_stats_sec['min']]
max_ts = [dtt2timestamp(dtt) for dtt in df_email_user_stats_sec['max']]
mode_ts = [dtt2timestamp(dtt) for dtt in df_email_user_stats_sec['mode']]
mean_ts = [dtt2timestamp(dtt) for dtt in df_email_user_stats_sec['mean']]


# In[31]:


df_email_user_stats_sec['min_ts'] = min_ts
df_email_user_stats_sec['max_ts'] = max_ts
df_email_user_stats_sec['mode_ts'] = mode_ts
df_email_user_stats_sec['mean_ts'] = mean_ts


# In[32]:


df_email_user_stats_sec


# In[33]:


df_email_user_stats_sec.drop(['min','max','mode','mean'], axis=1, inplace=True)


# In[ ]:





# In[34]:


df_email_user_stats_sec.dtypes


# In[35]:


email_stats = df_email_user_stats_sec.drop(['user_id'], axis = 1)

email_user_matrix = np.matrix(email_stats)


# In[36]:


email_user_matrix


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


# In[ ]:


forest = IsolationForest(bootstrap=False, contamination= 0.1 , max_features=1.0,
        max_samples='auto', n_estimators=100, n_jobs=1, random_state=0,
        verbose=0)
forest.fit(email_user_matrix)
email_ascore = forest.decision_function(email_user_matrix)
email_ascore[:10]


# In[ ]:


df_email_result = pd.DataFrame()
df_email_result['user_id'] = df_email_user_stats_sec['user_id']
df_email_result['ascore'] = email_ascore
print(df_email_result)


# In[ ]:


import pickle 
with open('../pkl/email_result.pkl','wb') as file :
  pickle.dump(forest, file)


# In[ ]:


#second Instance


# In[ ]:


# df_email_user_count = pd.read_csv(path+'user_email_count.csv')
# df_user_email_count


# In[ ]:




