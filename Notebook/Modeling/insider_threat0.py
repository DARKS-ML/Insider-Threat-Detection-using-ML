from Modules import sj_library  as sj
lib = sj.lb

import user_modules as um


# path = "/content/gdrive/MyDrive/Colab Notebooks/dataset/"
um.Show_infor('Users')
df_users_full = um.ReadCsv('users')
df_users1 = df_users_full[["user_id","functional_unit","department"]]
df_users = df_users1[df_users1.functional_unit == "2 - ResearchAndEngineering"]
df_users = df_users[(df_users.department != "1 - Research")]

df_users = pd.DataFrame(df_users)
df_users = df_users.dropna(axis=0)

um.Show_Shape('Users',df_users)



#read device
um.Show_infor('Device')
df_device = um.ReadCsv('device')
um.Show_Shape('Device',df_device)




#read file
um.Show_infor('File')
df_file = um.ReadCsv('file')
um.Show_Shape('File',df_file)




#read logon
um.Show_infor('Logon')
df_logon = um.ReadCsv('logon')
um.Show_Shape('LogoUsersn',df_logon)


# read psychometric
um.Show_infor('Psychometric')
df_psychometric = um.ReadCsv('psychometric')
df_psychometric = df_psychometric[['employee_name','user_id','O','C','E', 'A', 'N']]
um.Show_Shape('Psychometric',df_psychometric)



# ### Cleaning and merging
# clean user
df_users_clean = df_users.rename(columns= {'user_id':'user'}, inplace=False)
um.Show_Shape('clean User',df_users_clean)



# logon user clean
df_logon_users = pd.merge(df_logon, df_users_clean, on = 'user')
df_logon_users_clean = df_logon_users.drop(columns=['functional_unit', 'department'])
um.Show_Shape('LogonUser',df_logon_users_clean)


# device user clean
df_device_users = pd.merge(df_device, df_users_clean, on='user')
df_device_users_clean = df_device_users.drop(columns=['functional_unit','department'])
um.Show_Shape('Device user',df_device_users_clean)



# file user clean
df_file_users = pd.merge(df_file, df_users_clean, on = 'user')
df_file_users_clean = df_file_users.drop(columns = ['functional_unit', 'department'])
um.Show_Shape('File User',df_file_users_clean)


#[14]:


# psychometric users clean
df_psychometric_users = pd.merge(df_psychometric, df_users, on = 'user_id')
df_psychometric_users_clean = df_psychometric_users[['employee_name', 'user_id','O', 'C', 'E','A','N']]
um.Show_Shape('Psychometric Users clean',df_psychometric_users_clean)


# ### Section - 03 : Feature Engineering
#parsing date column to datetime
um.Convert_To_DT(df_logon_users_clean,'date')



# extract the time from date column add and store in a new column
df_logon_users_clean['time'] = df_logon_users_clean['date'].dt.time
print(df_logon_users_clean.head())




# Subsetting all records for 'Logon' activity
df_user_logon = df_logon_users_clean.loc[df_logon_users_clean['activity'] == 'Logon']
print(df_user_logon.head())



# add another column 'hour' for calculating mean Logon time
# time column is an object type and cannot be used for calculating mean
df_user_logon['hour'] = pd.to_datetime(df_user_logon['date'], format='%H:%M').dt.hour
print(df_user_logon.head())



# min and max login time of each user
#user_logon_stats_1 = user_logon.groupby
df_user_logon_stats = df_user_logon.groupby('user')['time'].agg([min,max]).reset_index()



df_user_logon_result = df_user_logon.groupby('user')['time'].agg(pd.Series.mode).reset_index()
print(df_user_logon.head())



# To handle the multi-modal situation
## value_counts arranged in descending order so you want to grab the index of the first row
df_logon_mode = df_user_logon.groupby('user')['time'].agg(lambda x: x.value_counts().index[0]).reset_index()
print(df_logon_mode.shape)



# calculate average (mean) logon time for each user

df_logon_mean = df_user_logon.groupby('user')['hour'].mean().reset_index()
df_logon_mean['hour'].dtype
df_logon_mean['hour'] = pd.to_datetime(df_logon_mean['hour'], format='%H').dt.time
print(df_logon_mean.head())
print(df_logon_mean.shape)



# Adding mode and mean data to user_logon_stats
df_user_logon_stats['mode'] = df_logon_mode['time']
df_user_logon_stats['mean'] = df_logon_mean['hour']
print(df_user_logon_stats.head())
print(df_user_logon_stats.shape)




#for logoff activity
df_user_logoff = df_logon_users_clean.loc[df_logon_users_clean['activity'] == 'Logoff']
print(df_user_logoff.shape)




um.Convert_To_DT(df_user_logoff,'date')
df_user_logoff['time'] = df_user_logoff['date'].dt.time
df_user_logoff['hour'] = pd.to_datetime(df_user_logoff['date'], format='%H:%M').dt.hour
print(df_user_logoff.head())




# min and max logoff time of each user
df_user_logoff_stats = df_user_logoff.groupby('user')['time'].agg([min,max]).reset_index()
print(df_user_logoff_stats.shape)




#mode logoff time for each user
df_user_logoff.groupby('user')['time'].agg(pd.Series.mode).reset_index()

## value_counts arranged in descending order so you want to grab the index of the first row
df_logoff_mode = df_user_logoff.groupby('user')['time'].agg(lambda x: x.value_counts().index[0]).reset_index()
print(df_logoff_mode.shape)



# Calculate mean logff
df_logoff_mean = df_user_logoff.groupby('user')['hour'].mean().reset_index()
df_logoff_mean['hour'] = df_logoff_mean['hour'].astype(int)
df_logoff_mean['hour'] = pd.to_datetime(df_logoff_mean['hour'], format='%H').dt.time
print(df_logoff_mean.shape)



# Adding mode and mean data to user_logoff_stats
df_user_logoff_stats['mode'] = df_logoff_mode['time']
df_user_logoff_stats['mean'] = df_logoff_mean['hour']
print(df_user_logoff_stats.shape)
print(df_user_logoff_stats.head())


# ### Device Feature Engineering


# extract time from date column and mutate a new column
df_device_users_clean['time'] = pd.to_datetime(df_device_users_clean['date']).dt.time
print(df_device_users_clean.shape)



# Subset records of Connect & Diconnect activities
df_device_conn = df_device_users_clean.loc[df_device_users_clean['activity'] == 'Connect']
df_device_disconn = df_device_users_clean.loc[df_device_users_clean['activity'] == 'Disconnect']
print(df_device_conn.shape)
print(df_device_disconn.shape)



# min max time for connect for each user
df_device_conn_stats = df_device_conn.groupby('user')['time'].agg([min, max]).reset_index()
print(df_device_conn_stats.head())



#
df_device_conn_stats_1 = df_device_conn.groupby('user')



## for mode: multimodel issue
df_conn_mode = df_device_conn.groupby('user')['time'].agg(lambda x: x.value_counts().index[0]).reset_index()



# Add extract hour and add a new column to device_conn in order to calculate average 'connect' time

df_device_conn['hour'] = pd.to_datetime(df_device_conn['date']).dt.hour
print(df_device_conn.head())



# mean 'connect'
df_conn_mean = df_device_conn.groupby('user')['hour'].mean().reset_index()
df_conn_mean['hour'] = df_conn_mean['hour'].astype(int)
df_conn_mean['hour'] = pd.to_datetime(df_conn_mean['hour'],format="%H").dt.time
print(df_conn_mean.shape)



# Add mode and mean to device_conn_stats
df_device_conn_stats['mode'] = df_conn_mode['time']
df_device_conn_stats['mean'] = df_conn_mean['hour']



#Device Disconnect Record
df_device_disconn_stats = df_device_disconn.groupby('user')['time'].agg([min,max]).reset_index()
print(df_device_disconn_stats.shape)



##correction below
df_dconn_mode = df_device_disconn.groupby('user')['time'].agg(lambda x: x.value_counts().index[0]).reset_index()
print(df_dconn_mode.shape)



# Extracting hour and adding hour column to calculate average time in device_disconn
df_device_disconn['hour'] = pd.to_datetime(df_device_disconn['date']).dt.hour
print(df_device_disconn.shape)



df_dconn_mean = df_device_disconn.groupby('user')['hour'].mean().reset_index()
df_dconn_mean['hour'] = df_dconn_mean['hour'].astype(int)
df_dconn_mean['hour'] = pd.to_datetime(df_dconn_mean['hour'], format='%H').dt.time
print(df_dconn_mean.shape)



# Adding mode and mean to device_disconn_stats

df_device_disconn_stats['mode'] = df_dconn_mode['time']
df_device_disconn_stats['mean'] = df_dconn_mean['hour']


# ### File


# Extract date (day) and add a new column to calculate max and mode files copy pey day for each user
# Each user has a normal number of files copy per day

df_file_users_clean['date2'] = pd.to_datetime(df_file_users_clean['date']).dt.date
print(df_file_users_clean.shape)



# to figure out the number of files transfered by a single user per day

df_files_per_day = df_file_users_clean.groupby(['user', 'date2']).size().reset_index()
print(df_files_per_day.shape)



df_files_per_day.rename(columns={0:'transfers_per_day'}, inplace=True)



# Max file transfers per day
df_files_max_per_day = df_files_per_day.groupby('user')['transfers_per_day'].agg(max).reset_index()
df_files_max_per_day.rename(columns={'transfers_per_day': "max_transfers_per_user"}, inplace=True)
print(df_files_max_per_day.shape)



##problem of multi-modal records
##to handle multi modal records, we do the following
df_files_mode_per_day = df_files_per_day.groupby('user')['transfers_per_day'].agg(lambda x: x.value_counts().index[0]).reset_index()
df_files_mode_per_day.rename(columns={'transfers_per_day': "mode_transfers_per_user"}, inplace=True)


#[48]:


# file copy stats for each user
df_files_stats = df_files_mode_per_day
print(df_files_stats.head())


#[49]:



df_files_stats_new = pd.DataFrame()
df_files_stats_new['user'] = df_files_stats['user']
df_files_stats_new['mode_trasfers_per_user'] = df_files_stats['mode_transfers_per_user']
df_files_stats_new['max_transfers_per_user'] = df_files_max_per_day['max_transfers_per_user']
print(df_files_stats_new.shape)


#[50]:


# #for user pc relation (tehi alxi lagne wala)
# df_user_pc = df_logon_users_clean.groupby(['user','pc',]).agg(pc_visits_per_user_total = pd.NamedAgg(column = 'pc', aggfunc = 'count'))
# df_user_pc.reset_index(level=1, inplace=True)
# df_user_pc.reset_index(level=0, inplace=True)
# df_user_pc['count'] = df_user_pc.groupby(['user'])['pc'].transform('nunique')
# print(df_user_pc.shape)


#[51]:


df_total_user_pc = df_logon_users_clean.groupby(['user','pc',]).agg(pc_visits_per_user_total = pd.NamedAgg(column = 'pc', aggfunc = 'count'))
print(df_total_user_pc.head())


#[52]:


df_total_user_pc.reset_index(level=1, inplace=True)
df_total_user_pc.reset_index(level=0, inplace=True)
print(df_total_user_pc.head())


#[53]:


print(len(df_total_user_pc.user.unique()))


#[54]:


df_total_user_pc['count'] = df_total_user_pc.groupby(['user'])['pc'].transform('nunique')
print(df_total_user_pc)


#[55]:


# user pc relation
df22 = df_total_user_pc[['user','count']]
print(df22.head())
print(df22.shape)
print(len(df22.user.unique()))


#[56]:


## dropping duplicate values
df22 = df22.drop_duplicates()


#[57]:


print(df22.head())
print(df22.shape)


#[58]:


df_user_pc = df22.copy()


#[59]:


print(df_user_pc.head())


#[60]:


print(df_user_pc['count'].unique())


# ###sider Threat Detection

# ### Anomaly Detection using Isolation Forests

#[61]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


#[62]:


# print(df_user_pc_count)
df_user_pc_count = np.array(df_user_pc['count'])


#[63]:


#Now trying to reshape with (-1, 1) . We have provided column as 1 but rows as unknown
df_user_pc_ct = df_user_pc_count.reshape(-1,1)


#[64]:


### Modeling user_pc relationship using Isolation Forests
forest = IsolationForest(bootstrap=False, contamination=0.1,max_features=1.0,
                         max_samples='auto',n_estimators=100, n_jobs=1, random_state=None,
                         verbose=0)
forest.fit(df_user_pc_ct)


#[ ]:


# predictions
# user_pc_predict = forest.predict(df_user_pc_ct)
# pred_outliers = forest.predict(?)
# scores = forest.score_samples(df_user_pc_ct)


#[ ]:


# # new, 'normal' observations ----
# print("Accuracy:", list(y_pred_test).count(1)/y_pred_test.shape[0])
# # Accuracy: 0.93
# # outliers ----
# print("Accuracy:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])
# # Accuracy: 0.96


#[65]:


print(df_user_pc_ct.shape)


#[66]:


graph_a_score = forest.decision_function(df_user_pc_ct)
print(graph_a_score[1:10])


#[67]:


graph_result = pd.DataFrame()
graph_result['user'] = df_user_pc['user']
graph_result['ascore'] = graph_a_score
print(graph_result)


#[68]:


#outliers.describe()
#graph_result.sort_values('ascore')
outliers = graph_result.loc[graph_result['ascore'] < 0]
print(outliers.head())
print(outliers.shape)


# #### Anomaly detection for log on/log off using Isolation Forests

#[69]:


print(df_user_logon_stats.head())


#[70]:


## sklearn algorithms do not take dates as input parameters.
## hence need to convert the time object into numerical values.
# Function to convert datetime 'time' to time in seconds

def dtt2timestamp(dtt):
  time_in_sec = (dtt.hour*60 + dtt.minute) * 60 + dtt.second
  return time_in_sec


#[71]:


#user_logon_stats_sec
df_user_logon_stats_sec = df_user_logon_stats


#[72]:


min_ts = [dtt2timestamp(dtt) for dtt in df_user_logon_stats_sec['min']]
max_ts = [dtt2timestamp(dtt) for dtt in df_user_logon_stats_sec['max']]
mode_ts = [dtt2timestamp(dtt) for dtt in df_user_logon_stats_sec['mode']]
mean_ts = [dtt2timestamp(dtt) for dtt in df_user_logon_stats_sec['mean']]


#[73]:


df_user_logon_stats_sec['min_ts'] = min_ts
df_user_logon_stats_sec['max_ts'] = max_ts
df_user_logon_stats_sec['mode_ts'] = mode_ts
df_user_logon_stats_sec['mean_ts'] = mean_ts


#[74]:


df_user_logon_stats_sec.drop(['min', 'max','mode','mean'], axis=1)


#[75]:


df_user_logon_stats_sec.drop(['min','max','mode','mean'], axis=1)


# ### Logoff

#[76]:


print(df_user_logoff_stats.head())


#[77]:


# Make copy of user logoff stats
df_user_logoff_stats_sec = df_user_logoff_stats


#[78]:


min_ts = [dtt2timestamp(dtt) for dtt in df_user_logoff_stats_sec['min']]
max_ts = [dtt2timestamp(dtt) for dtt in df_user_logoff_stats_sec['max']]
mode_ts = [dtt2timestamp(dtt) for dtt in df_user_logoff_stats_sec['mode']]
mean_ts = [dtt2timestamp(dtt) for dtt in df_user_logoff_stats_sec['mean']]


#[79]:


df_user_logoff_stats_sec['min_ts'] = min_ts
df_user_logoff_stats_sec['max_ts'] = max_ts
df_user_logoff_stats_sec['mode_ts'] = mode_ts
df_user_logoff_stats_sec['mean_ts'] = mean_ts


#[80]:


df_user_logoff_stats_sec.drop(['min', 'max','mode','mean'], axis=1)


#[81]:


# combined logon/logoff data for IForest input

df_log_on_off_stats = pd.DataFrame()

df_log_on_off_stats['user'] = df_user_logon_stats_sec['user']

df_log_on_off_stats['on_min_ts'] = df_user_logon_stats_sec['min_ts']
df_log_on_off_stats['on_max_ts'] = df_user_logon_stats_sec['max_ts']
df_log_on_off_stats['on_mode_ts'] = df_user_logon_stats_sec['mode_ts']
df_log_on_off_stats['on_mean_ts'] = df_user_logon_stats['mean_ts']

df_log_on_off_stats['off_min_ts'] = df_user_logon_stats_sec['min_ts']
df_log_on_off_stats['off_max_ts'] = df_user_logon_stats_sec['max_ts']
df_log_on_off_stats['off_mode_ts'] = df_user_logon_stats_sec['mode_ts']
df_log_on_off_stats['off_mean_ts'] = df_user_logon_stats_sec['mean_ts']

print(df_log_on_off_stats.head())
print(df_log_on_off_stats.shape)


# #### Modeling logon/logoff relationship using Isolation Forests

#[82]:


#put array
log_stats = df_log_on_off_stats.drop(['user'], axis = 1)
log_stats_matrix = np.matrix(log_stats)
print(log_stats_matrix)


#[83]:


forest = IsolationForest(bootstrap=False, contamination= 0.1 , max_features=1.0,
        max_samples='auto', n_estimators=100, n_jobs=1, random_state=None,
        verbose=0)
forest.fit(log_stats_matrix)
log_ascore = forest.decision_function(log_stats_matrix)
print(log_ascore[:10])
print(log_ascore.shape)


#[84]:


df_user_log_result = pd.DataFrame()
df_user_log_result['user'] = df_user_logoff_stats_sec['user']
df_user_log_result['ascore'] = log_ascore
print(df_user_log_result.head())
print(df_user_log_result.shape)


#[85]:


##OUTLIERS
#user_log_result.sort_values('ascore')
outliers_user = df_user_log_result.loc[df_user_log_result['ascore'] < 0]
print(outliers_user.head())


# #### Model with removable device and file transfer stats as ip

#[86]:


print(df_device_conn_stats.head())


#[87]:


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

print(df_device_conn_stats_sec.head())
print(df_device_conn_stats_sec.shape)


# ### Device with disconnect Activity

#[88]:


print(df_device_disconn_stats.head())


#[89]:


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

print(df_device_disconn_stats_sec.head())
print(df_device_disconn_stats_sec.shape)


# #### Files and Device

#[90]:


# Combine all the removable media (device) parameters
df_device_full = pd.DataFrame()
df_device_full['user'] = df_device_conn_stats['user']


#[91]:


# connect stats
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


#[92]:


print(df_device_full.head())
print(df_device_full.shape)


# ### Model fitting

#[93]:


device_full_matrix = df_device_full.drop(['user'],axis=1)
device_params = np.matrix(device_full_matrix)
print(device_params[:10])


#[94]:


# fit the model
forest = IsolationForest(bootstrap=False, contamination=0.1, max_features=1.0,
        max_samples='auto', n_estimators=100, n_jobs=1, random_state=None,
        verbose=0)
forest.fit(device_params)


#[95]:


## Anomaly Score

dev_file_ascore = forest.decision_function(device_params)
print(dev_file_ascore)


#[96]:


# Save the result
df_device_file_full_result = pd.DataFrame()

df_device_file_full_result['user'] = df_device_full['user']
df_device_file_full_result['ascore'] = dev_file_ascore
print(df_device_file_full_result.head())


#[97]:


#device_file_full_result.sort_values('ascore')
df_device_file_full_result_sorted = df_device_file_full_result.loc[df_device_file_full_result['ascore'] < 0]
print(df_device_file_full_result_sorted)


# #### Anomaly detection with psychometric data

#[98]:


print(df_psychometric_users_clean.head())


#[99]:


# fit the model
#put array
psychometric_matrix = df_psychometric_users_clean.drop(['user_id', 'employee_name'], axis = 1)
psychometric_params = np.matrix(psychometric_matrix)
print(psychometric_params)


#[100]:


forest = IsolationForest(bootstrap=False, contamination=0.1, max_features=1.0,
        max_samples='auto', n_estimators=100, n_jobs=1, random_state=None,
        verbose=0)
forest.fit(psychometric_params)


#[101]:


# anomaly score
psych_ascore = forest.decision_function(psychometric_params)
print(psych_ascore[:10])


#[102]:


#psych_ascore.shape
df_psychometric_result = pd.DataFrame()

df_psychometric_result['user'] = df_psychometric_users_clean['user_id']
df_psychometric_result['ascore'] = psych_ascore
print(df_psychometric_result.head())
print(df_psychometric_result.shape)


#[103]:


#OUTLIERS
#psychometric_result.sort_values('ascore')
df_psychometric_result_outlier = df_psychometric_result.loc[df_psychometric_result['ascore'] < 0]
print(df_psychometric_result_outlier.head())
print(df_psychometric_result_outlier.shape)


# #### Model with some of input features combined

#[104]:


df = pd.merge(df_log_on_off_stats, df_user_pc, on='user')
print(df.head())


#[105]:


df_1 = pd.merge(df, df_psychometric, left_on = 'user', right_on = 'user_id')
print(df_1.head())
print(df_1.shape)


#[106]:


df_final = df_1.drop(['employee_name', 'user_id'], axis=1)
print(df_final.head())
print(df_final.shape)


#[107]:


# Make copy
df_all_parameters = df_final


#[108]:


# Model Fitting
#input array
df_all_parameters_input = df_all_parameters.drop(['user'], axis = 1)
print(df_all_parameters_input.head())
print(df_all_parameters_input.shape)


#[109]:


forest = IsolationForest(bootstrap=False, contamination=0.1, max_features=1.0,
        max_samples='auto', n_estimators=100, n_jobs=1, random_state=None,
        verbose=0)
forest.fit(df_all_parameters_input)


#[110]:


#anomaly score
all_parameters_ascore = forest.decision_function(df_all_parameters_input)
print(all_parameters_ascore[:10])


#[111]:


#psych_ascore.shape
df_all_parameters_result = pd.DataFrame()

df_all_parameters_result['user'] = df_final['user']
df_all_parameters_result['ascore'] = all_parameters_ascore
print(df_all_parameters_result.head())


#[112]:


#OUTLIERS
df_all_parameters_result_outlier = df_all_parameters_result.loc[df_all_parameters_result['ascore'] < 0]
print(df_all_parameters_result_outlier.head())
print(df_all_parameters_result_outlier.shape)


# ### Visualization

#[113]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#%matplotlib inline


#[114]:


f, ax = plt.subplots(figsize = (20,10))
x_col='user'
y_col = 'ascore'
sns.set_theme(style="darkgrid")
sns.pointplot(ax=ax,x=x_col,y=y_col,data=df_all_parameters_result,color='purple')
sns.pointplot(ax=ax,x=x_col,y=y_col,data=df_user_log_result,color='grey')
sns.pointplot(ax=ax,x=x_col,y=y_col,data=df_psychometric_result,color='brown')
sns.pointplot(ax=ax,x=x_col,y=y_col,data=df_device_file_full_result,color='darkorange')

ax.legend(handles=ax.lines[::len(df_all_parameters_result)+1], labels=["All","Graph","Logon/Logoff","Psychometric","Removable Media"])
#ax.set_xtickslabels(rotation = 45)
ax.axhline(0, ls='-')
ax.set_title('Anomaly score for different set of parameters ', size = 20)
plt.rcParams["axes.labelsize"] = 25
plt.xticks(rotation = 45, fontsize = 10)
plt.yticks(fontsize = 10)
plt.show()


#[115]:


# file transfer per user
f, ax = plt.subplots(figsize = (25,15))
x_col='user'

sns.pointplot(ax=ax,x=x_col,y='mode_trasfers_per_user',data=df_files_stats_new, color='orange')
sns.pointplot(ax=ax,x=x_col,y='max_transfers_per_user',data=df_files_stats_new,color='blue')

ax.legend(handles=ax.lines[::len(df_files_stats)+1], labels=["mode", "max"], fontsize = 20)

ax.set_title('File transfers per user', size = 30)
plt.rcParams["axes.labelsize"] = 25
plt.ylabel("Number of files")
plt.xticks(rotation = 45, fontsize = 10)
plt.yticks(fontsize = 10)
# plt.legend(fontsize=20)
plt.show()


#[ ]:





#[116]:


df_user_log_result.hist(bins = 15)


#[117]:


df_device_file_full_result.hist(bins = 15)


#[118]:


df_psychometric_result.hist()


#[119]:


df_all_parameters_result.hist()


#[120]:


## All parameters combined
df_all_parameters_result.loc[df_all_parameters_result['ascore'] < 0].hist()


#[121]:


df_threat_users_all_params = df_all_parameters_result.loc[df_all_parameters_result['ascore'] < -0.065]
print(df_threat_users_all_params)


#[122]:


df_all_parameters_result = df_all_parameters[df_all_parameters.user.isin(df_threat_users_all_params.user)]
print(df_all_parameters_result.head())
print(df_all_parameters_result.shape)


#[123]:


print(df_all_parameters.on_max_ts.median())


# ### Device and File

#[124]:


df_device_file_full_result.loc[df_device_file_full_result['ascore'] < 0].hist()


#[125]:


df_threat_users_device_file = df_device_file_full_result.loc[df_device_file_full_result['ascore'] <= -0]
print(df_threat_users_device_file.head())


#[126]:


df_device_full_result = df_device_full[df_device_full.user.isin(df_threat_users_device_file.user)]
print(df_device_full.head())
print(f"File Mode Mean: "+ str(df_device_full.file_mode.mean()))
print(f"File Max Mean: "+ str(df_device_full.file_max.mean()))


# ### Logon and Logoff

#[127]:


df_user_log_result.loc[df_user_log_result['ascore'] < 0].hist()


#[128]:


df_threat_users_log = df_user_log_result.loc[df_user_log_result['ascore'] <= -0.04]
print(df_threat_users_log)
print(df_threat_users_log.shape)


#[129]:


print(df_log_on_off_stats[df_log_on_off_stats.user.isin(df_threat_users_log.user)])

print(f'Max Mean:'+ str(df_log_on_off_stats.on_max_ts.mean()))
print(f'Max ts Median:'+ str(df_log_on_off_stats.on_max_ts.median()))


#[ ]:





#[130]:


#psychometric_users_clean
df_psychometric_result.loc[df_psychometric_result['ascore'] < 0].hist()


#[131]:


df_threat_psycho = df_psychometric_result.loc[df_psychometric_result['ascore'] <= -0.04]
print(df_threat_psycho.head())
print(df_threat_psycho.shape)


#[132]:


df_psychometric_users_clean_result = df_psychometric_users_clean[df_psychometric_users_clean.user_id.isin(df_threat_psycho.user)]
print(df_psychometric_users_clean_result.head())
print(df_psychometric_users_clean_result.shape)


#[135]:


df_all_parameters.isna().sum()


#[136]:


df_all_parameters.columns


#[137]:



def anomaly_scatterplot(y,z, xlabtitle, ylabtitle):
    x = df_all_parameters['user']
    df = pd.concat([x, y, z], axis=1)
    df_matrix = df.drop(['user'], axis = 1)
    forest = IsolationForest(bootstrap=False, contamination=0.1, max_features=1.0,
        max_samples='auto', n_estimators=100, n_jobs=1, random_state=None,
        verbose=0)
    forest.fit(df_matrix)
    ascore = forest.decision_function(df_matrix)
    result = pd.DataFrame()
    result['user'] = x
    result_full = pd.merge(result, df, on = 'user')
    result_full['ascore'] = ascore
    result_full['Anomaly'] = np.where(result_full['ascore'] > 0, 1, -1)
    sns.set(rc={'figure.figsize':(12,10)})
    plot = sns.scatterplot(data=result_full, x = y, y = z, s = 125, hue = result_full['Anomaly'], palette=['red','green'])
    plt.xlabel(xlabtitle, fontsize = 18)
    plt.ylabel(ylabtitle, fontsize = 18)
    plt.legend(bbox_to_anchor=(1.01, 0.5),borderaxespad=0, title = "Anomaly" )
    plt.title('Anomaly Scatterplot', fontsize=20)
    plt.tight_layout()






    return plot


#[138]:


anomaly_scatterplot(df_all_parameters['on_min_ts'], df_all_parameters['off_min_ts'], 'Minimum Login Time (in sec)', 'Minimum Logoff Time (in sec)')
