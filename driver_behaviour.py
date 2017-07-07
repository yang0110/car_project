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

# t61_data1=pd.read_pickle('D:car/data/new_10_car_data_1/'+dir_t61[0])#9
# t1_data1=pd.read_pickle('D:car/data/new_10_car_data_1/'+dir_t1[1])##189
# t61_data2=pd.read_pickle('D:car/data/new_10_car_data_2/9-2016-01-01--2017-01-01.pickle')
# t1_data2=pd.read_pickle('D:car/data/new_10_car_data_2/189-2016-01-01--2017-01-01.pickle')

dir_t66_55=dir_set[(dir_set2=='55')]
frame=pd.read_pickle('D:car/data/new_10_car_data_1/'+dir_t66_55[0])#55
frame.index=pd.to_datetime(frame.index)
new_shape=0
for i, j in enumerate(dir_t66_55[1:]):
	print (i, len(dir_t66[1:]))
	data=pd.read_pickle('D:car/data/new_10_car_data_1/'+dir_t66_55[i])
	data.index=pd.to_datetime(data.index)
	shape=data.shape[0]
	new_shape+=shape 
	frame=frame.append(data)
	print (frame.shape) 


acc_data=frame.iloc[:,[36,37]]
acc_data.columns=['acc', 'brake']

speed_data=pd.read_pickle('D:car/data/new_10_car_data_2/55-2016-01-01--2017-01-01.pickle')

speed_data=speed_data.iloc[:,2:5]
speed_data.columns=['longitude','latitude','speed']
speed_data=speed_data.iloc[:-33,:]# remove rows with timestamp is wrong format
#####
acc_data.index=pd.to_datetime(acc_data.index)
speed_data.index=pd.to_datetime(speed_data.index)

##add new features 
acc_data['time']=acc_data.index
acc_data['acc_on_off']=0
acc_data.acc_on_off[acc_data.acc>0]=1
acc_data['brake_on_off']=0
acc_data.brake_on_off[acc_data.brake>0]=1

###add status flag to speed and acc dataset 
acc_data['acc']=acc_data['acc'].apply(float)
acc_data['brake']=acc_data['brake'].apply(float)
acc_data['status']=0
acc_data['status'][(acc_data['acc']>0) | (acc_data['brake']>0)]=1
##
speed_data['speed']=speed_data['speed'].apply(float)
speed_data['run_status']=0
speed_data['run_status'][speed_data['speed']>0]=1
speed_data['stop_status']=0
speed_data['stop_status'][speed_data['speed']==0]=1
##combine speed and acc dataset 

####
def runs_of_ones_list(bits):
  return [sum(g) for b, g in itertools.groupby(bits) if b==1]


zero_bits=speed_data['stop_status'].values
speed_zeros=runs_of_ones_list(zero_bits)

one_bits=speed_data['run_status'].values
speed_ones=runs_of_ones_list(one_bits)


fig=plt.plot(figsize=(10,10))
plt.plot(speed_zeros[:10000])
plt.title('speed stop length')
plt.show()

fig=plt.plot(figsize=(10,10))
plt.plot(speed_ones[:1000])
plt.title('speed  length')
plt.show()


fig=plt.plot(figsize=(10,10))
plt.hist(speed_zeros, bins=np.linspace(min(speed_zeros), max(speed_zeros),20))
plt.show()

fig=plt.plot(figsize=(10,10))
plt.hist(speed_ones, bins=np.linspace(min(speed_ones), max(speed_ones),20))
plt.show()

fig=plt.plot(figsize=(10,10))
plt.plot(speed_data.speed.iloc[:5000])
plt.title('speed')
plt.show()

fig=plt.plot(figsize=(10,10))
plt.plot(acc_data.acc.iloc[:5000])
plt.title('acc')
plt.show()

### stop how long 
speed_data_index=speed_data.index.values
stop_time_length=np.zeros(len(speed_zeros))
time=0
for i,j in enumerate(speed_zeros):
	stop_time_length[i]=float((speed_data_index[time+j]-speed_data_index[time]))/float((60*60*(10**9))) ## hour
	time+=j 


fig=plt.plot(figsize=(10,10))
plt.plot(stop_time_length[np.where(stop_time_length<1)][:10000]*60)
plt.title('stop   length')
plt.ylabel('time (minute)')
plt.show()


### average the speed 
speed_data['rolling_mean_speed']=pd.rolling_mean(speed_data.speed, 20)
speed_data=speed_data.fillna(0)
speed_data['rolling_mean_speed'].iloc[:5000].plot()
plt.show()

speed_data['rolling_run_status']=0
speed_data['rolling_run_status'][speed_data['rolling_mean_speed']!=0.0]=1

bits=speed_data.rolling_run_status.values 
run_length=runs_of_ones_list(bits)


speed_data['rolling_stop_status']=0
speed_data['rolling_stop_status'][speed_data['rolling_mean_speed']==0.0]=1
bits=speed_data.rolling_stop_status.values
stop_length=runs_of_ones_list(bits)

print (sum(run_length)+sum(stop_length))
print (speed_data.shape)



speed_data['timestamp']=speed_data.index
speed_data_index=speed_data.index.values
speed_data['status']=0
time=0
for i in range(len(stop_length)):
	time+=stop_length[i]
	if run_length[i]<100:
		speed_data['status'][time:time+run_length[i]]=0
		time+=run_length[i]
	else:
		speed_data['status'][time:time+run_length[i]]=1
		time+=run_length[i]

speed_data['hour']=speed_data.timestamp.dt.hour
speed_data['status'][(speed_data.hour>=21) | (speed_data.hour<=5)]=0
speed_data['status'][speed_data.status==1]=np.mean(speed_data.speed)

fig=plt.plot(figsize=(10,10))
plt.plot(speed_data['speed'][1000:5000],color='b', label='speed')
plt.plot(speed_data['status'][1000:5000][speed_data.status!=0], '.',color='r', label='running')
plt.plot(speed_data['status'][1000:5000][speed_data.status==0], '.',color='k', label='stop')
plt.legend(loc='best',fontsize=12)
plt.ylabel('speed', fontsize=12)
plt.xlabel('time', fontsize=12)
plt.title('bus T66 Car 55 runing segments', fontsize=15)
plt.savefig('D:car/data/figures/bus_t66_car_55_running_segments.png')
plt.show()

speed_data['status'][speed_data['status']!=0]=1
run_length2=runs_of_ones_list(speed_data.status.values)
print (len(run_length2))

speed_data['stop_status']=0
speed_data['stop_status'][speed_data.status==0]=1
stop_length2=runs_of_ones_list(speed_data['stop_status'].values)
print (len(stop_length2))

from slugify import slugify
time=0
for i in range(len(stop_length2)):
	time+=int(stop_length2[i])
	run_data=speed_data.iloc[time:time+int(run_length2[i]), :]
	time+=int(run_length2[i])
	run_data.to_csv('D:/car/data/bus_t66_car_55_run_segments/bus_t66_car_55_%s_%s'%(slugify(str(run_data.index[0])), slugify(str(run_data.index[-1]))))
###########83
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

speed_data=pd.read_pickle('D:car/data/new_10_car_data_2/83-2016-01-01--2017-01-01.pickle')
speed_data=speed_data.iloc[:-1000,:].copy()
speed_data=speed_data.iloc[:,2:5]
speed_data.columns=['longitude','latitude','speed']
#####
acc_data.index=pd.to_datetime(acc_data.index)
speed_data.index=pd.to_datetime(speed_data.index)

##add new features 
acc_data['time']=acc_data.index
acc_data['acc_on_off']=0
acc_data.acc_on_off[acc_data.acc>0]=1
acc_data['brake_on_off']=0
acc_data.brake_on_off[acc_data.brake>0]=1

###add status flag to speed and acc dataset 
acc_data['acc']=acc_data['acc'].apply(float)
acc_data['brake']=acc_data['brake'].apply(float)
acc_data['status']=0
acc_data['status'][(acc_data['acc']>0) | (acc_data['brake']>0)]=1
##
speed_data['speed']=speed_data['speed'].apply(float)
speed_data['run_status']=0
speed_data['run_status'][speed_data['speed']>0]=1
speed_data['stop_status']=0
speed_data['stop_status'][speed_data['speed']==0]=1
##combine speed and acc dataset 

####
def runs_of_ones_list(bits):
  return [sum(g) for b, g in itertools.groupby(bits) if b==1]


zero_bits=speed_data['stop_status'].values
speed_zeros=runs_of_ones_list(zero_bits)

one_bits=speed_data['run_status'].values
speed_ones=runs_of_ones_list(one_bits)


fig=plt.plot(figsize=(10,10))
plt.plot(speed_zeros[:10000])
plt.title('speed stop length')
plt.show()

fig=plt.plot(figsize=(10,10))
plt.plot(speed_ones[:1000])
plt.title('speed  length')
plt.show()


fig=plt.plot(figsize=(10,10))
plt.hist(speed_zeros, bins=np.linspace(min(speed_zeros), max(speed_zeros),20))
plt.show()

fig=plt.plot(figsize=(10,10))
plt.hist(speed_ones, bins=np.linspace(min(speed_ones), max(speed_ones),20))
plt.show()

fig=plt.plot(figsize=(10,10))
plt.plot(speed_data.speed.iloc[:5000])
plt.title('speed')
plt.show()

fig=plt.plot(figsize=(10,10))
plt.plot(acc_data.acc.iloc[:5000])
plt.title('acc')
plt.show()

### stop how long 
speed_data_index=speed_data.index.values
stop_time_length=np.zeros(len(speed_zeros))
time=0
for i,j in enumerate(speed_zeros):
	stop_time_length[i]=float((speed_data_index[time+j]-speed_data_index[time]))/float((60*60*(10**9))) ## hour
	time+=j 


fig=plt.plot(figsize=(10,10))
plt.plot(stop_time_length[np.where(stop_time_length<1)][:10000]*60)
plt.title('stop   length')
plt.ylabel('time (minute)')
plt.show()


### average the speed 
speed_data['rolling_mean_speed']=pd.rolling_mean(speed_data.speed, 20)
speed_data=speed_data.fillna(0)
speed_data['rolling_mean_speed'].iloc[:5000].plot()
plt.show()

speed_data['rolling_run_status']=0
speed_data['rolling_run_status'][speed_data['rolling_mean_speed']!=0.0]=1

bits=speed_data.rolling_run_status.values 
run_length=runs_of_ones_list(bits)


speed_data['rolling_stop_status']=0
speed_data['rolling_stop_status'][speed_data['rolling_mean_speed']==0.0]=1
bits=speed_data.rolling_stop_status.values
stop_length=runs_of_ones_list(bits)

print (sum(run_length)+sum(stop_length))
print (speed_data.shape)



speed_data['timestamp']=speed_data.index
speed_data_index=speed_data.index.values
speed_data['status']=0
time=0
for i in range(len(stop_length)):
	time+=stop_length[i]
	if run_length[i]<100:
		speed_data['status'][time:time+run_length[i]]=0
		time+=run_length[i]
	else:
		speed_data['status'][time:time+run_length[i]]=1
		time+=run_length[i]

speed_data['hour']=speed_data.timestamp.dt.hour
speed_data['status'][(speed_data.hour>=21) | (speed_data.hour<=5)]=0
speed_data['status'][speed_data.status==1]=np.mean(speed_data.speed)

fig=plt.plot(figsize=(10,10))
plt.plot(speed_data['speed'][1000:5000],color='b', label='speed')
plt.plot(speed_data['status'][1000:5000][speed_data.status!=0], '.',color='r', label='running')
plt.plot(speed_data['status'][1000:5000][speed_data.status==0], '.',color='k', label='stop')
plt.legend(loc='best',fontsize=12)
plt.ylabel('speed', fontsize=12)
plt.xlabel('time', fontsize=12)
plt.title('bus T66 Car 83 runing segments', fontsize=15)
plt.savefig('D:car/data/figures/bus_t66_car_83_running_segments.png')
plt.show()

speed_data['status'][speed_data['status']!=0]=1
run_length2=runs_of_ones_list(speed_data.status.values)
print (len(run_length2))

speed_data['stop_status']=0
speed_data['stop_status'][speed_data.status==0]=1
stop_length2=runs_of_ones_list(speed_data['stop_status'].values)
print (len(stop_length2))

from slugify import slugify
time=0
for i in range(len(run_length2)):
	time+=int(stop_length2[i])
	run_data=speed_data.iloc[time:time+int(run_length2[i]), :]
	time+=int(run_length2[i])
	run_data.to_csv('D:/car/data/bus_t66_car_83_run_segments/bus_t66_car_83_%s_%s'%(slugify(str(run_data.index[0])), slugify(str(run_data.index[-1]))))



######91
dir_t66_95=dir_set[(dir_set2=='95')]
frame=pd.read_pickle('D:car/data/new_10_car_data_1/'+dir_t66_95[0])#55
frame.index=pd.to_datetime(frame.index)
new_shape=0
for i, j in enumerate(dir_t66_95[1:]):
	print (i, len(dir_t66[1:]))
	data=pd.read_pickle('D:car/data/new_10_car_data_1/'+dir_t66_95[i])
	data.index=pd.to_datetime(data.index)
	shape=data.shape[0]
	new_shape+=shape 
	frame=frame.append(data)
	print (frame.shape) 


acc_data=frame.iloc[:,[36,37]]
acc_data.columns=['acc', 'brake']

speed_data=pd.read_pickle('D:car/data/new_10_car_data_2/95-2016-01-01--2017-01-01.pickle')
speed_data=speed_data.iloc[:-1000,:].copy()
speed_data=speed_data.iloc[:,2:5]
speed_data.columns=['longitude','latitude','speed']
#####
acc_data.index=pd.to_datetime(acc_data.index)
speed_data.index=pd.to_datetime(speed_data.index)

##add new features 
acc_data['time']=acc_data.index
acc_data['acc_on_off']=0
acc_data.acc_on_off[acc_data.acc>0]=1
acc_data['brake_on_off']=0
acc_data.brake_on_off[acc_data.brake>0]=1

###add status flag to speed and acc dataset 
acc_data['acc']=acc_data['acc'].apply(float)
acc_data['brake']=acc_data['brake'].apply(float)
acc_data['status']=0
acc_data['status'][(acc_data['acc']>0) | (acc_data['brake']>0)]=1
##
speed_data['speed']=speed_data['speed'].apply(float)
speed_data['run_status']=0
speed_data['run_status'][speed_data['speed']>0]=1
speed_data['stop_status']=0
speed_data['stop_status'][speed_data['speed']==0]=1
##combine speed and acc dataset 

####
def runs_of_ones_list(bits):
  return [sum(g) for b, g in itertools.groupby(bits) if b==1]


zero_bits=speed_data['stop_status'].values
speed_zeros=runs_of_ones_list(zero_bits)

one_bits=speed_data['run_status'].values
speed_ones=runs_of_ones_list(one_bits)


fig=plt.plot(figsize=(10,10))
plt.plot(speed_zeros[:10000])
plt.title('speed stop length')
plt.show()

fig=plt.plot(figsize=(10,10))
plt.plot(speed_ones[:1000])
plt.title('speed  length')
plt.show()


fig=plt.plot(figsize=(10,10))
plt.hist(speed_zeros, bins=np.linspace(min(speed_zeros), max(speed_zeros),20))
plt.show()

fig=plt.plot(figsize=(10,10))
plt.hist(speed_ones, bins=np.linspace(min(speed_ones), max(speed_ones),20))
plt.show()

fig=plt.plot(figsize=(10,10))
plt.plot(speed_data.speed.iloc[:5000])
plt.title('speed')
plt.show()

fig=plt.plot(figsize=(10,10))
plt.plot(acc_data.acc.iloc[:5000])
plt.title('acc')
plt.show()

### stop how long 
speed_data_index=speed_data.index.values
stop_time_length=np.zeros(len(speed_zeros))
time=0
for i,j in enumerate(speed_zeros):
	stop_time_length[i]=float((speed_data_index[time+j]-speed_data_index[time]))/float((60*60*(10**9))) ## hour
	time+=j 


fig=plt.plot(figsize=(10,10))
plt.plot(stop_time_length[np.where(stop_time_length<1)][:10000]*60)
plt.title('stop   length')
plt.ylabel('time (minute)')
plt.show()


### average the speed 
speed_data['rolling_mean_speed']=pd.rolling_mean(speed_data.speed, 20)
speed_data=speed_data.fillna(0)
speed_data['rolling_mean_speed'].iloc[:5000].plot()
plt.show()

speed_data['rolling_run_status']=0
speed_data['rolling_run_status'][speed_data['rolling_mean_speed']!=0.0]=1

bits=speed_data.rolling_run_status.values 
run_length=runs_of_ones_list(bits)


speed_data['rolling_stop_status']=0
speed_data['rolling_stop_status'][speed_data['rolling_mean_speed']==0.0]=1
bits=speed_data.rolling_stop_status.values
stop_length=runs_of_ones_list(bits)

print (sum(run_length)+sum(stop_length))
print (speed_data.shape)



speed_data['timestamp']=speed_data.index
speed_data_index=speed_data.index.values
speed_data['status']=0
time=0
for i in range(len(stop_length)):
	time+=stop_length[i]
	if run_length[i]<100:
		speed_data['status'][time:time+run_length[i]]=0
		time+=run_length[i]
	else:
		speed_data['status'][time:time+run_length[i]]=1
		time+=run_length[i]

speed_data['hour']=speed_data.timestamp.dt.hour
speed_data['status'][(speed_data.hour>=21) | (speed_data.hour<=5)]=0
speed_data['status'][speed_data.status==1]=np.mean(speed_data.speed)

fig=plt.plot(figsize=(10,10))
plt.plot(speed_data['speed'][1000:5000],color='b', label='speed')
plt.plot(speed_data['status'][1000:5000][speed_data.status!=0], '.',color='r', label='running')
plt.plot(speed_data['status'][1000:5000][speed_data.status==0], '.',color='k', label='stop')
plt.legend(loc='best',fontsize=12)
plt.ylabel('speed', fontsize=12)
plt.xlabel('time', fontsize=12)
plt.title('bus T66 Car 95 runing segments', fontsize=15)
plt.savefig('D:car/data/figures/bus_t66_car_95_running_segments.png')
plt.show()

speed_data['status'][speed_data['status']!=0]=1
run_length2=runs_of_ones_list(speed_data.status.values)
print (len(run_length2))

speed_data['stop_status']=0
speed_data['stop_status'][speed_data.status==0]=1
stop_length2=runs_of_ones_list(speed_data['stop_status'].values)
print (len(stop_length2))

from slugify import slugify
time=0
for i in range(len(run_length2)):
	time+=int(stop_length2[i])
	run_data=speed_data.iloc[time:time+int(run_length2[i]), :]
	time+=int(run_length2[i])
	run_data.to_csv('D:/car/data/bus_t66_car_95_run_segments/bus_t66_car_95_%s_%s'%(slugify(str(run_data.index[0])), slugify(str(run_data.index[-1]))))
