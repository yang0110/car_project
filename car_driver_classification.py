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
from slugify import slugify




### read acc_brake dataset 
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
acc_data['timestamp']=acc_data.index
acc_data=acc_data.resample('1S', how='mean')
acc_data['acc_onoff']=0
acc_data['acc_onoff'][acc_data.acc>0]=1
acc_data['brake_onoff']=0
acc_data['brake_onoff'][acc_data.brake>0]=1

speed_dir_set=os.listdir('D:/car/data/bus_t66_car_95_run_segments')
## read the 1st dataset 
for i in range(len(speed_dir_set)):
	print (i)

	speed_data=pd.read_csv('D:/car/data/bus_t66_car_95_run_segments/'+speed_dir_set[i])
	## remove unnecessary columns
	speed_data=speed_data.drop('0', axis=1)
	speed_data.index=pd.to_datetime(speed_data.timestamp)

	speed_data=speed_data.resample('1S', how='mean')
	speed_data.index=pd.to_datetime(speed_data.index)
	## select acc_brake dataset according the timestamp of speed data 
	time_start=str(speed_data.index[0])
	time_end=str(speed_data.index[-1])

	acc_small_data=acc_data[time_start:time_end]
	### combine acc_brake data and speed_data 
	com_data=pd.concat([speed_data, acc_small_data], axis=0)
	com_data=com_data.drop(['hour','latitude','longitude','rolling_mean_speed','rolling_run_status','rolling_stop_status','run_status','stop_status', 'status'], axis=1)
	mean_com_data=com_data.resample('5min',how='mean')
	mean_columns=mean_com_data.columns+'_mean'
	mean_com_data.columns=mean_columns

	sum_com_data=com_data.resample('5min',how='sum')
	sum_columns=sum_com_data.columns+'_sum'
	sum_com_data.columns=sum_columns

	new_com_data=pd.concat([mean_com_data, sum_com_data],axis=1)
	new_com_data=new_com_data.drop(['acc_onoff_mean','brake_onoff_mean','brake_sum','speed_sum','acc_sum'],axis=1)
	index=new_com_data.index
	new_com_data.to_csv('D:Dev/car_project/data/t66_car95_5min_data/_%s_%s'%(slugify(str(index[0])), slugify(str(index[-1]))))