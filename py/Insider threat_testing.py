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
path = '../dataset new/'


# In[4]:


df_users_full = pd.read_csv(path + "users.csv")
df_users1 = df_users_full[["user_id","functional_unit","department"]]
df_users = df_users1[df_users1.functional_unit == "2 - ResearchAndEngineering"]
df_users = df_users[(df_users.department != "1 - Research")]
df_users = pd.DataFrame(df_users)


# In[ ]:





# In[5]:


df_users = df_users.dropna(axis = 0)
df_users.info()


# In[6]:


df_device = pd.read_csv(path + "device.csv")


# In[7]:


df_file = pd.read_csv(path + "file.csv")


# In[8]:


df_logon = pd.read_csv(path + "logon.csv")


# In[9]:


df_psychometric = pd.read_csv(path + "psychometric.csv")
df_psychometric = df_psychometric[['employee_name','user_id','O','C','E', 'A', 'N']]


# In[10]:


df_users_clean = df_users.rename(columns= {'user_id':'user'}, inplace=False)


# In[11]:


df_logon_users = pd.merge(df_logon, df_users_clean, on = 'user')
df_logon_users_clean = df_logon_users.drop(columns=['functional_unit', 'department'])


# In[12]:


df_device_users = pd.merge(df_device, df_users_clean, on='user')
df_device_users_clean = df_device_users.drop(columns=['functional_unit','department'])


# In[13]:


df_device_users = pd.merge(df_device, df_users_clean, on = 'user')
df_device_users_clean = df_device_users.drop(columns = ['functional_unit', 'department'])


# In[14]:


df_file_users = pd.merge(df_file, df_users_clean, on = 'user')
df_file_users_clean = df_file_users.drop(columns = ['functional_unit', 'department'])


# In[15]:


df_psychometric_users = pd.merge(df_psychometric, df_users, on = 'user_id')
df_psychometric_users_clean = df_psychometric_users[['employee_name', 'user_id','O', 'C', 'E','A','N']]


# In[ ]:





# In[16]:


df_logon_users_clean['date'] = pd.to_datetime(df_logon_users_clean['date'])


# In[17]:


df_logon_users_clean['time'] = df_logon_users_clean['date'].dt.time


# In[18]:


df_user_logon = df_logon_users_clean.loc[df_logon_users_clean['activity'] == 'Logon']


# In[19]:


df_user_logon['hour'] = pd.to_datetime(df_user_logon['date'], format='%H:%M').dt.hour


# In[20]:


df_user_logon_stats = df_user_logon.groupby('user')['time'].agg([min,max]).reset_index()


# In[21]:


df_logon_mode = df_user_logon.groupby('user')['time'].agg(lambda x: x.value_counts().index[0]).reset_index()


# In[ ]:





# In[22]:


df_logon_mean = df_user_logon.groupby('user')['hour'].mean().reset_index()
df_logon_mean['hour'].dtype
df_logon_mean['hour'] = pd.to_datetime(df_logon_mean['hour'], format='%H').dt.time


# In[23]:


df_user_logon_stats['mode'] = df_logon_mode['time']
df_user_logon_stats['mean'] = df_logon_mean['hour']


# In[ ]:





# In[24]:


df_user_logoff = df_logon_users_clean.loc[df_logon_users_clean['activity'] == 'Logoff']


# In[25]:


df_user_logoff['date'] = pd.to_datetime(df_user_logoff['date'])
df_user_logoff['time'] = df_user_logoff['date'].dt.time


# In[26]:



df_user_logoff['hour'] = pd.to_datetime(df_user_logoff['date'], format='%H:%M').dt.hour


# In[27]:


df_user_logoff_stats = df_user_logoff.groupby('user')['time'].agg([min,max]).reset_index()


# In[28]:


df_user_logoff.groupby('user')['time'].agg(pd.Series.mode).reset_index()


# In[29]:


df_logoff_mode = df_user_logoff.groupby('user')['time'].agg(lambda x: x.value_counts().index[0]).reset_index()


# In[30]:


df_logoff_mean = df_user_logoff.groupby('user')['hour'].mean().reset_index()
df_logoff_mean['hour'] = df_logoff_mean['hour'].astype(int)
df_logoff_mean['hour'] = pd.to_datetime(df_logoff_mean['hour'], format='%H').dt.time


# In[31]:


df_user_logoff_stats['mode'] = df_logoff_mode['time']
df_user_logoff_stats['mean'] = df_logoff_mean['hour']


# In[ ]:





# In[32]:


df_device_users_clean['time'] = pd.to_datetime(df_device_users_clean['date']).dt.time


# In[33]:


df_device_conn = df_device_users_clean.loc[df_device_users_clean['activity'] == 'Connect']
df_device_disconn = df_device_users_clean.loc[df_device_users_clean['activity'] == 'Disconnect']


# In[34]:


df_device_conn_stats = df_device_conn.groupby('user')['time'].agg([min, max]).reset_index()


# In[35]:


df_device_conn_stats_1 = df_device_conn.groupby('user')


# In[36]:


df_conn_mode = df_device_conn.groupby('user')['time'].agg(lambda x: x.value_counts().index[0]).reset_index()


# In[37]:


df_device_conn['hour'] = pd.to_datetime(df_device_conn['date']).dt.hour


# In[38]:


df_conn_mean = df_device_conn.groupby('user')['hour'].mean().reset_index()
df_conn_mean['hour'] = df_conn_mean['hour'].astype(int)
df_conn_mean['hour'] = pd.to_datetime(df_conn_mean['hour'],format="%H").dt.time


# In[39]:


df_device_conn_stats['mode'] = df_conn_mode['time']
df_device_conn_stats['mean'] = df_conn_mean['hour']


# In[ ]:





# In[40]:


df_device_disconn_stats = df_device_disconn.groupby('user')['time'].agg([min,max]).reset_index()


# In[41]:


df_dconn_mode = df_device_disconn.groupby('user')['time'].agg(lambda x: x.value_counts().index[0]).reset_index()


# In[42]:


df_device_disconn['hour'] = pd.to_datetime(df_device_disconn['date']).dt.hour


# In[43]:


df_dconn_mean = df_device_disconn.groupby('user')['hour'].mean().reset_index()
df_dconn_mean['hour'] = df_dconn_mean['hour'].astype(int)
df_dconn_mean['hour'] = pd.to_datetime(df_dconn_mean['hour'], format='%H').dt.time


# In[44]:


df_dconn_mean


# In[45]:


df_device_disconn_stats['mode'] = df_dconn_mode['time']
df_device_disconn_stats['mean'] = df_dconn_mean['hour']


# In[46]:


df_file_users_clean['date2'] = pd.to_datetime(df_file_users_clean['date']).dt.date


# In[ ]:





# In[47]:


df_files_per_day = df_file_users_clean.groupby(['user', 'date2']).size().reset_index()


# In[ ]:





# In[48]:


df_files_per_day.rename(columns={0:'transfers_per_day'}, inplace=True)


# In[49]:


df_files_max_per_day = df_files_per_day.groupby('user')['transfers_per_day'].agg(max).reset_index()
df_files_max_per_day.rename(columns={'transfers_per_day': "max_transfers_per_user"}, inplace=True)


# In[50]:


df_files_mode_per_day = df_files_per_day.groupby('user')['transfers_per_day'].agg(lambda x: x.value_counts().index[0]).reset_index()
df_files_mode_per_day.rename(columns={'transfers_per_day': "mode_transfers_per_user"}, inplace=True)


# In[51]:


df_files_mode_per_day


# In[52]:


df_files_stats = df_files_mode_per_day


# In[ ]:





# In[53]:


df_files_stats_new = pd.DataFrame()
df_files_stats_new['user'] = df_files_stats['user']
df_files_stats_new['mode_trasfers_per_user'] = df_files_stats['mode_transfers_per_user']
df_files_stats_new['max_transfers_per_user'] = df_files_max_per_day['max_transfers_per_user']


# In[54]:


df_user_pc = df_logon_users_clean.groupby(['user','pc',]).agg(pc_visits_per_user_total = pd.NamedAgg(column = 'pc', aggfunc = 'count')).reset_index()
df_user_pc['count'] = df_user_pc.groupby(['user'])['pc'].transform('nunique')
df_user_pc = df_user_pc.drop(['pc', 'pc_visits_per_user_total'], axis=1)
df_user_pc = df_user_pc.drop_duplicates()


# In[55]:


df_user_pc


# In[56]:


# import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


# In[57]:


df_user_pc_count = np.array(df_user_pc['count'])


# In[58]:


df_user_pc_ct = df_user_pc_count.reshape(-1,1)


# In[59]:


import pickle
model = pickle.load(open('../pkl/user_pc_ct.pkl', 'rb'))
graph_a_score = model.decision_function(df_user_pc_ct)
threat = model.predict(df_user_pc_ct)


# In[60]:


#user pc
#forest = IsolationForest(bootstrap=False, contamination=0.1,max_features=1.0,
#                          max_samples='auto',n_estimators=100, n_jobs=1, random_state=None,
#                          verbose=0)
# # forest.fit(df_user_pc_ct)


# In[61]:


# import pickle 
# with open('user_pc_ct.pkl','wb') as file :
#   pickle.dump(forest, file)


# In[62]:


# graph_a_score = forest.decision_function(df_user_pc_ct)
# print(graph_a_score[1:10])


# In[63]:


graph_result = pd.DataFrame()
graph_result['user'] = df_user_pc['user']
graph_result['ascore'] = graph_a_score
graph_result['threat']= threat
graph_result.to_csv('../output/user_pc_ct.csv', index=False)
graph_result


# In[64]:


outliers = graph_result.loc[graph_result['ascore'] < 0]
print(outliers)


# In[65]:


def dtt2timestamp(dtt):
  time_in_sec = (dtt.hour*60 + dtt.minute) * 60 + dtt.second
  return time_in_sec


# In[66]:


df_user_logon_stats_sec = df_user_logon_stats


# In[ ]:





# In[67]:


min_ts = [dtt2timestamp(dtt) for dtt in df_user_logon_stats_sec['min']]
max_ts = [dtt2timestamp(dtt) for dtt in df_user_logon_stats_sec['max']]
mode_ts = [dtt2timestamp(dtt) for dtt in df_user_logon_stats_sec['mode']]
mean_ts = [dtt2timestamp(dtt) for dtt in df_user_logon_stats_sec['mean']]


# In[68]:


df_user_logon_stats_sec['min_ts'] = min_ts
df_user_logon_stats_sec['max_ts'] = max_ts
df_user_logon_stats_sec['mode_ts'] = mode_ts
df_user_logon_stats_sec['mean_ts'] = mean_ts


# In[ ]:





# In[69]:


df_user_logon_stats_sec.drop(['min','max','mode','mean'], axis=1)


# In[70]:


df_user_logoff_stats_sec = df_user_logoff_stats


# In[ ]:





# In[71]:


min_ts = [dtt2timestamp(dtt) for dtt in df_user_logoff_stats_sec['min']] 
max_ts = [dtt2timestamp(dtt) for dtt in df_user_logoff_stats_sec['max']]
mode_ts = [dtt2timestamp(dtt) for dtt in df_user_logoff_stats_sec['mode']]
mean_ts = [dtt2timestamp(dtt) for dtt in df_user_logoff_stats_sec['mean']]


# In[72]:


df_user_logoff_stats_sec['min_ts'] = min_ts
df_user_logoff_stats_sec['max_ts'] = max_ts
df_user_logoff_stats_sec['mode_ts'] = mode_ts
df_user_logoff_stats_sec['mean_ts'] = mean_ts


# In[73]:


df_user_logoff_stats_sec.drop(['min', 'max','mode','mean'], axis=1)


# In[74]:


df_log_on_off_stats = pd.DataFrame()

df_log_on_off_stats['user'] = df_user_logon_stats_sec['user']
df_log_on_off_stats['on_min_ts'] = df_user_logon_stats_sec['min_ts']
df_log_on_off_stats['on_max_ts'] = df_user_logon_stats_sec['max_ts']
df_log_on_off_stats['on_mode_ts'] = df_user_logon_stats_sec['mode_ts']
df_log_on_off_stats['on_mean_ts'] = df_user_logon_stats_sec['mean_ts']
df_log_on_off_stats['off_min_ts'] = df_user_logon_stats_sec['min_ts']
df_log_on_off_stats['off_max_ts'] = df_user_logon_stats_sec['max_ts']
df_log_on_off_stats['off_mode_ts'] = df_user_logon_stats_sec['mode_ts']
df_log_on_off_stats['off_mean_ts'] = df_user_logon_stats_sec['mean_ts']


# In[75]:


df_log_on_off_stats.dtypes


# In[76]:


log_stats = df_log_on_off_stats.drop(['user'], axis = 1)
log_stats_matrix = np.matrix(log_stats)
print(log_stats_matrix)


# In[77]:


df_log_on_off_stats.columns


# In[78]:


# 

model2 = pickle.load(open('../pkl/log_stats_matrix.pkl', 'rb'))
log_ascore = model2.decision_function(log_stats_matrix)
threat = model2.predict(log_stats_matrix)


# In[79]:


# # counting the values
# df22 = pd.Series(forest.predict(log_stats_matrix))
# # df22 = df22.map({1:0, -1:1})
# # print(df22.value_counts())


# In[80]:


# import pickle 
# with open('log_stats_matrix.pkl','wb') as file :
#   pickle.dump(forest, file)


# In[81]:


df_user_log_result = pd.DataFrame()
df_user_log_result['user'] = df_user_logoff_stats_sec['user']
df_user_log_result['ascore'] = log_ascore
df_user_log_result['threat'] = threat
df_user_log_result.to_csv('../output/user_log_result.csv', index=False)
print(df_user_log_result)


# In[82]:


df_user_log_result.loc[df_user_log_result['ascore'] < 0]


# In[83]:


df_device_conn_stats_sec = df_device_conn_stats
con_min_ts = [dtt2timestamp(dtt) for dtt in df_device_conn_stats_sec['min']]
con_max_ts = [dtt2timestamp(dtt) for dtt in df_device_conn_stats_sec['max']]
con_mode_ts = [dtt2timestamp(dtt) for dtt in df_device_conn_stats_sec['mode']]
con_mean_ts = [dtt2timestamp(dtt) for dtt in df_device_conn_stats_sec['mean']]

df_device_conn_stats_sec['min_ts'] = con_min_ts
df_device_conn_stats_sec['max_ts'] = con_max_ts
df_device_conn_stats_sec['mode_ts'] = con_mode_ts
df_device_conn_stats_sec['mean_ts'] = con_mean_ts
df_device_conn_stats_sec = df_device_conn_stats_sec.drop(['min', 'max','mode','mean'], axis=1)


# In[84]:


df_device_disconn_stats_sec = df_device_conn_stats
discon_min_ts = [dtt2timestamp(dtt) for dtt in df_device_disconn_stats_sec['min']]
discon_max_ts = [dtt2timestamp(dtt) for dtt in df_device_disconn_stats_sec['max']]
discon_mode_ts = [dtt2timestamp(dtt) for dtt in df_device_disconn_stats_sec['mode']]
discon_mean_ts = [dtt2timestamp(dtt) for dtt in df_device_disconn_stats_sec['mean']]

df_device_disconn_stats_sec['min_ts'] = discon_min_ts
df_device_disconn_stats_sec['max_ts'] = discon_max_ts
df_device_disconn_stats_sec['mode_ts'] = discon_mode_ts
df_device_disconn_stats_sec['mean_ts'] = discon_mean_ts
df_device_disconn_stats_sec = df_device_disconn_stats_sec.drop(['min', 'max','mode','mean'], axis=1)


# In[85]:


df_device_full = pd.DataFrame()
df_device_full['user'] = df_device_conn_stats['user']


# In[86]:


df_device_full['con_min_ts'] = df_device_conn_stats_sec['min_ts']
df_device_full['con_max_ts'] = df_device_conn_stats_sec['max_ts']
df_device_full['con_mode_ts'] = df_device_conn_stats_sec['mode_ts']
df_device_full['con_mean_ts'] = df_device_conn_stats_sec['mean_ts']

# disconnect stats
df_device_full['dcon_min_ts'] = df_device_disconn_stats_sec['min_ts']
df_device_full['dcon_max_ts'] = df_device_disconn_stats_sec['max_ts']
df_device_full['dcon_mode_ts'] = df_device_disconn_stats_sec['mode_ts']
df_device_full['dcon_mean_ts'] = df_device_disconn_stats_sec['mean_ts']

# files per day stats
df_device_full['file_mode'] = df_files_stats_new['mode_trasfers_per_user']
df_device_full['file_max'] = df_files_stats_new['max_transfers_per_user']


# In[87]:


device_full_matrix = df_device_full.drop(['user'],axis=1)
device_params = np.matrix(device_full_matrix)
device_params[:10]


# In[88]:


df_device_full.columns


# In[89]:


# # device (-user)
# forest = IsolationForest(bootstrap=False, contamination=0.1, max_features=1.0,
#         max_samples='auto', n_estimators=100, n_jobs=1, random_state=None,
#         verbose=0)

# forest.fit(device_params)


# In[90]:


# import pickle 
# with open('device_params.pkl','wb') as file :
#   pickle.dump(forest, file)


# In[91]:


# 

model3 = pickle.load(open('../pkl/device_params.pkl', 'rb'))
log_ascore = model3.decision_function(device_params)
threat = model3.predict(device_params)


# In[92]:


df_device_file_full_result = pd.DataFrame()

df_device_file_full_result['user'] = df_device_full['user']
df_device_file_full_result['ascore'] = log_ascore
df_device_file_full_result['threat'] = threat

df_device_file_full_result.to_csv('../output/device_file_full_result.csv', index=False)


# In[93]:


df_device_file_full_result.loc[df_device_file_full_result['ascore'] < 0] 


# In[94]:


psychometric_matrix = df_psychometric_users_clean.drop(['user_id', 'employee_name'], axis = 1)
psychometric_params = np.matrix(psychometric_matrix)


# In[95]:


# # psycho ds 
# forest = IsolationForest(bootstrap=False, contamination=0.1, max_features=1.0,
#         max_samples='auto', n_estimators=100, n_jobs=1, random_state=None,
#         verbose=0)
# forest.fit(psychometric_params)


# In[96]:


# import pickle 
# with open('psychometric_params.pkl','wb') as file :
#   pickle.dump(forest, file)


# In[98]:


# 

model4 = pickle.load(open('../pkl/psychometric_params.pkl', 'rb'))
# df2=pd.DataFrame()
log_ascore = model4.decision_function(psychometric_params)
threat = model4.predict(psychometric_params)


# In[99]:


df_psychometric_result = pd.DataFrame()

df_psychometric_result['user'] = df_psychometric_users_clean['user_id']
df_psychometric_result['ascore'] = log_ascore
df_psychometric_result['threat']= threat
df_psychometric_result.to_csv('../output/psychometric_result.csv', index=False)
df_psychometric_result


# In[100]:


df_psychometric_result.loc[df_psychometric_result['ascore'] < 0]


# In[ ]:





# In[101]:


df = pd.merge(df_log_on_off_stats, df_user_pc, on='user')


# In[ ]:





# In[102]:


df_1 = pd.merge(df, df_psychometric, left_on = 'user', right_on = 'user_id')


# In[103]:


df_1.head()


# In[104]:


df_final = df_1.drop(['employee_name', 'user_id'], axis=1)


# In[105]:


df_all_parameters = df_final


# In[106]:


df_all_parameters_input = df_all_parameters.drop(['user'], axis = 1)


# In[107]:


print(df_all_parameters_input)


# In[108]:


# # psycho logon ds
# forest = IsolationForest(bootstrap=False, contamination=0.1, max_features=1.0,
#         max_samples='auto', n_estimators=100, n_jobs=1, random_state=None,
#         verbose=0)
# forest.fit(df_all_parameters_input)


# In[109]:


# import pickle 
# with open('all_parameters_input.pkl','wb') as file :
#   pickle.dump(forest, file)


# In[111]:


# 

model5 = pickle.load(open('../pkl/all_parameters_input.pkl', 'rb'))
# df2=pd.DataFrame()
log_ascore = model5.decision_function(df_all_parameters_input)
threat = model5.predict(df_all_parameters_input)


# In[112]:


df_all_parameters_result = pd.DataFrame()

df_all_parameters_result['user'] = df_final['user']
df_all_parameters_result['ascore'] =log_ascore
df_all_parameters_result['threat'] = threat
print(df_all_parameters_result)
df_all_parameters_result.to_csv('../output/all_parameters_result.csv', index=False)


# In[ ]:





# In[ ]:




