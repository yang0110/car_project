import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import date
import holidays
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from matplotlib import cm 
from sklearn.metrics import silhouette_samples
from scipy.stats import pearsonr
from matplotlib import collections as matcoll
from scipy.signal import argrelextrema
import datetime
import os
from numpy import diff
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
from pandas.tseries.offsets import *
from sklearn.mixture import GMM
from sklearn.neighbors.kde import KernelDensity
from scipy.stats.distributions import norm
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics import silhouette_score
import itertools 

dir_set=os.listdir('D:car/data/new_10_car_data_1')
dir_set2=np.array([x.split('-')[0] for x in dir_set])
dir_set=np.array(dir_set)
dir_9=dir_set[dir_set2=='9']

unique_bus=np.unique(dir_set2)[:-1]
#['16', '189', '191', '195', '55', '83', '9', '91', '95', '99']
t61=['9','13']
t66=['55','83','91','95','99']
t1=['189','191','195']
dir_t61=dir_set[(dir_set2=='9') | (dir_set2=='13')]
dir_t66=dir_set[(dir_set2=='55')|(dir_set2=='83')|(dir_set2=='91')|(dir_set2=='95')|(dir_set2=='99')]
dir_t1=dir_set[(dir_set2=='189')|(dir_set2=='191')|(dir_set2=='195')]



dir_t66_83=dir_set[(dir_set2=='83')]
frame=pd.read_pickle('D:car/data/new_10_car_data_1/'+dir_t66_83[0])#55
frame.index=pd.to_datetime(frame.index)
new_shape=0
for i, j in enumerate(dir_t66_83[1:]):
	print (i, len(dir_t66[1:]))
	data=pd.read_pickle('D:car/data/new_10_car_data_1/'+dir_t66_83[i])
	data.index=pd.to_datetime(data.index)
	shape=data.shape[0]
	new_shape+=shape 
	frame=frame.append(data)
	print (frame.shape) 


acc_data=frame.iloc[:,[36,37]]
acc_data.columns=['acc', 'brake']
acc_data.index=pd.to_datetime(acc_data.index)

##add new features 
acc_data['time']=acc_data.index
acc_data['acc_on_off']=0
acc_data.acc_on_off[acc_data.acc>0]=1
acc_data['brake_on_off']=0
acc_data.brake_on_off[acc_data.brake>0]=1
acc_data['acc']=acc_data['acc'].apply(float)
acc_data['brake']=acc_data['brake'].apply(float)

acc_data['run_status']=0
acc_data['run_status'][(acc_data['acc']>0) | (acc_data['brake']>0)]=1

def runs_of_ones_list(bits):
  return [sum(g) for b, g in itertools.groupby(bits) if b==1]

run_bits=acc_data['run_status'].values
acc_ones=runs_of_ones_list(run_bits)

fig=plt.plot(figsize=(10,10))
plt.plot(acc_data['acc'])
plt.show()

fig=plt.plot(figsize=(10,10))
plt.plot(acc_data['brake'])
plt.show()


fig=plt.plot(figsize=(10,10))
plt.plot(acc_ones[:1000])
plt.show()

### average the speed 
acc_data['rolling_mean_acc']=pd.rolling_mean(acc_data.acc, 20)
acc_data=acc_data.fillna(0)
acc_data['rolling_mean_acc'].iloc[:5000].plot()

fig=plt.plot(figsize=(10,10))
plt.plot(acc_data['rolling_mean_acc'])
plt.show()


acc_data['rolling_run_status']=0
acc_data['rolling_run_status'][acc_data['rolling_mean_acc']!=0.0]=1

bits=acc_data.rolling_run_status.values 
run_length=runs_of_ones_list(bits)


acc_data['rolling_stop_status']=0
acc_data['rolling_stop_status'][acc_data['rolling_mean_acc']==0.0]=1
bits=acc_data.rolling_stop_status.values
stop_length=runs_of_ones_list(bits)

print (sum(run_length)+sum(stop_length))
print (acc_data.shape)



acc_data['timestamp']=acc_data.index
acc_data_index=acc_data.index.values
acc_data['status']=0
time=0
for i in range(len(stop_length)):
	time+=stop_length[i]
	acc_data['status'][time:time+run_length[i]]=1
	time+=run_length[i]

acc_data['hour']=acc_data.timestamp.dt.hour
acc_data['status'][(acc_data.hour>=21) | (acc_data.hour<=5)]=0
acc_data['status'][acc_data.status==1]=np.mean(acc_data.acc)

fig=plt.plot(figsize=(10,10))
plt.plot(acc_data['acc'][:10000],color='b', label='acc')
plt.plot(acc_data[acc_data.status!=0]['status'][:10000], '.',color='r', label='running')
plt.plot(acc_data[acc_data.status==0]['status'][:10000], '.',color='k', label='stop')
plt.legend(loc='best',fontsize=12)
plt.ylabel('acc', fontsize=12)
plt.xlabel('time', fontsize=12)
plt.title('bus T66 Car 83 acc runing segments', fontsize=15)
plt.savefig('D:car/data/figures/bus_t66_car_83_acc_running_segments.png')
plt.show()

acc_data['status'][acc_data['status']!=0]=1
run_length2=runs_of_ones_list(acc_data.status.values)
print (len(run_length2))

acc_data['stop_status']=0
acc_data['stop_status'][acc_data.status==0]=1
stop_length2=runs_of_ones_list(acc_data['stop_status'].values)
print (len(stop_length2))

from slugify import slugify
time=0
for i in range(len(stop_length2)):
	time+=int(stop_length2[i])
	run_data=acc_data.iloc[time:time+int(run_length2[i]), :]
	time+=int(run_length2[i])
	run_data.to_csv('D:/car/data/bus_t66_acc_run_segments/bus_t66_car_83_acc_run_segments/bus_t66_car_83_%s_%s'%(slugify(str(run_data.index[0])), slugify(str(run_data.index[-1]))))
