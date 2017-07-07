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

t61_data1=pd.read_pickle('D:car/data/new_10_car_data_1/'+dir_t61[0])#9
t66_data1=pd.read_pickle('D:car/data/new_10_car_data_1/'+dir_t66[1])#55
t1_data1=pd.read_pickle('D:car/data/new_10_car_data_1/'+dir_t1[1])##189

t61_data2=pd.read_pickle('D:car/data/new_10_car_data_2/9-2016-01-01--2017-01-01.pickle')
t66_data2=pd.read_pickle('D:car/data/new_10_car_data_2/55-2016-01-01--2017-01-01.pickle')
t1_data2=pd.read_pickle('D:car/data/new_10_car_data_2/189-2016-01-01--2017-01-01.pickle')

### t66
pos_speed_data=t66_data2.iloc[:,2:5]
pos_speed_data.columns=['longitude','latitude','speed']
acc_brake_data=t66_data1[[36,37]].copy()
acc_brake_data.columns=['acc','brake']
time_str=pos_speed_data.index[0][:7]
new_pos_speed=pos_speed_data.iloc[:acc_brake_data.shape[0],:]
new_pos_speed=pos_speed_data.iloc[:30000,:]

acc_brake_data.index=pd.to_datetime(acc_brake_data.index)
new_pos_speed.index=pd.to_datetime(new_pos_speed.index)

##add new features 
acc_brake_data['time']=acc_brake_data.index
acc_brake_data['acc_on_off']=0
acc_brake_data.acc_on_off[acc_brake_data.acc>0]=1
acc_brake_data['brake_on_off']=0
acc_brake_data.brake_on_off[acc_brake_data.brake>0]=1
## 1 min sample
e_minu1=acc_brake_data.resample('1min',how='mean')
e_minu2=acc_brake_data.resample('1min',how='sum')
e_minu1.columns=['acc_mean','brake_mean','acc_on_off_mean','brake_on_off_mean']
e_minu2.columns=['acc_sum','brake_sum','acc_on_off_sum','brake_on_off_sum']				
e_minu3=pd.concat([e_minu1,e_minu2],axis=1)
new_acc_brake=e_minu3[['acc_mean','brake_mean','acc_on_off_sum','brake_on_off_sum']]
new_acc_brake=new_acc_brake.dropna()

new_pos_speed=new_pos_speed.astype(float).resample('1min',how='mean')
new_pos_speed.columns=['longitude','latitude','speed_mean']
new_pos_speed=new_pos_speed.dropna()

t66_data=pd.concat([new_pos_speed,new_acc_brake],axis=1).dropna()
t66_new_data=t66_data[['speed_mean','acc_mean','brake_mean','acc_on_off_sum','brake_on_off_sum']]
mms=MinMaxScaler()
norm_input=mms.fit_transform(t66_new_data)
n_clusters=7
km=KMeans(n_clusters=n_clusters,init='k-means++', n_init=200).fit(norm_input)
labels=km.labels_.reshape((len(km.labels_),1))
t66_new_data['label']=labels+1
t66_data['label']=labels+1
t66_data.to_csv('D:car/data/t66_data_with_label')
features=['speed_mean', 'acc_mean', 'brake_mean', 'acc_on_off_sum',
       		'brake_on_off_sum', 'label']

t66_new_data=t66_new_data[features]
features=list(t66_new_data.columns)
fig,ax=plt.subplots(5,1,figsize=(15,15),sharex=True)
ax=ax.ravel()
for i, j in enumerate(features[:-1]):
	sns.violinplot(x='label',y=j,points=100, data=t66_new_data,
					ax=ax[i],palette="Set2",split=True,scale='count',inner='quartile')
	ax[i].set_xlabel('Road Type',fontsize=15)
	ax[i].set_xticklabels(list(np.unique(t66_new_data.label.values)))
	ax[i].set_ylabel('%s'%(j),fontsize=15)
plt.savefig('D:car/data/t66_violinplot_of_clusters.png')
plt.show()

####t1
pos_speed_data=t1_data2.iloc[:,2:5]
pos_speed_data.columns=['longitude','latitude','speed']
acc_brake_data=t1_data1[[36,37]].copy()
acc_brake_data.columns=['acc','brake']
time_str=pos_speed_data.index[0][:7]
new_pos_speed=pos_speed_data.iloc[:acc_brake_data.shape[0],:]
new_pos_speed=pos_speed_data.iloc[:30000,:]

acc_brake_data.index=pd.to_datetime(acc_brake_data.index)
new_pos_speed.index=pd.to_datetime(new_pos_speed.index)
##add new features 
acc_brake_data['time']=acc_brake_data.index
acc_brake_data['acc_on_off']=0
acc_brake_data.acc_on_off[acc_brake_data.acc>0]=1
acc_brake_data['brake_on_off']=0
acc_brake_data.brake_on_off[acc_brake_data.brake>0]=1
## 1 min sample
e_minu1=acc_brake_data.resample('1min',how='mean')
e_minu2=acc_brake_data.resample('1min',how='sum')
e_minu1.columns=['acc_mean','brake_mean','acc_on_off_mean','brake_on_off_mean']
e_minu2.columns=['acc_sum','brake_sum','acc_on_off_sum','brake_on_off_sum']				
e_minu3=pd.concat([e_minu1,e_minu2],axis=1)
new_acc_brake=e_minu3[['acc_mean','brake_mean','acc_on_off_sum','brake_on_off_sum']]
new_acc_brake=new_acc_brake.dropna()

new_pos_speed=new_pos_speed.astype(float).resample('1min',how='mean')
new_pos_speed.columns=['longitude','latitude','speed_mean']
new_pos_speed=new_pos_speed.dropna()

t1_data=pd.concat([new_pos_speed,new_acc_brake],axis=1).dropna()
t1_new_data=t1_data[['speed_mean','acc_mean','brake_mean','acc_on_off_sum','brake_on_off_sum']]

mms=MinMaxScaler()
norm_input=mms.fit_transform(t1_new_data)
# n_clusters=7
# km=KMeans(n_clusters=n_clusters,init='k-means++', n_init=200).fit(norm_input)
# labels=km.labels_.reshape((len(km.labels_),1))
# t1_new_data['label']=labels+1
# t1_data['label']=labels+1
t1_data.to_csv('D:car/data/t1_data_with_label')
features=['speed_mean', 'acc_mean', 'brake_mean', 'acc_on_off_sum',
       		'brake_on_off_sum', 'label']

t1_new_data=t1_new_data[features]
features=list(t1_new_data.columns)
fig,ax=plt.subplots(5,1,figsize=(15,15),sharex=True)
ax=ax.ravel()
for i, j in enumerate(features[:-1]):
	sns.violinplot(x='label',y=j,points=100, data=t1_new_data,
                      ax=ax[i],palette="Set2",split=True,scale='count',inner='quartile')
	ax[i].set_xlabel('Road Type',fontsize=15)
	ax[i].set_xticklabels(list(np.unique(t1_new_data.label.values)))
	ax[i].set_ylabel('%s'%(j),fontsize=15)
plt.savefig('D:car/data/t1_violinplot_of_clusters.png')
plt.show()


### bus stops
t66_bus_stop=pd.read_csv('D:car/data/t66_bus_stop.csv',header=None)
t1_bus_stop=pd.read_csv('D:car/data/t1_bus_stop.csv',header=None)

longi=np.array([float(x.split(',')[0]) for x in t66_bus_stop.iloc[:,0]])
lati=np.array([float(x.split(',')[1]) for x in t66_bus_stop.iloc[:,0]])
new_t66_bus_stop=np.hstack((longi.reshape(len(longi),1),lati.reshape(len(lati),1)))
t66_stops=pd.DataFrame(new_t66_bus_stop,columns=['longitude','latitude'])

longi=np.array([float(x.split(',')[0]) for x in t1_bus_stop.iloc[:,0]])
lati=np.array([float(x.split(',')[1]) for x in t1_bus_stop.iloc[:,0]])
new_t1_bus_stop=np.hstack((longi.reshape(len(longi),1),lati.reshape(len(lati),1)))
t1_stops=pd.DataFrame(new_t1_bus_stop,columns=['longitude','latitude'])

###T66
mean_la=np.mean(t66_stops.latitude.values)
mean_lo=np.mean(t66_stops.longitude.values)


color_bar_ticklabels=['Clear Way','Lightly Jam','Start from Cross-Sections','Heavily Jam','Start From Bus Stops',
						'still','Others']
color=['lawngreen','pink','c','r','b','k','purple']

time=np.arange(6,19,1)
google_baidu_la=-0.0053289999999996951
google_baidu_lo=-0.0062369999999987158

google_bus_la=-0.0022963333333336777
google_bus_lo=0.0061813333333446963

for i,j in enumerate (time):
	print (i,j)
	if j<10:
		time_str='2016-07-25 0'+str(j)	
	else:
		time_str='2016-07-25 '+str(j)
	if j<=12:
		am_pm='am'
	else:
		am_pm='pm'

	new_combine=t66_data[time_str]
	gmap = gmplot.GoogleMapPlotter(mean_la,mean_lo,16)
	gmap.scatter(t66_stops.latitude.values+google_baidu_la, t66_stops.longitude.values+google_baidu_lo, 'gray', size=100, marker=False)
	for ii in np.unique(new_combine.label.values):
		la_data=new_combine['latitude'][new_combine.label==ii].values
		lo_data=new_combine['longitude'][new_combine.label==ii].values
		col=color[int(ii-1)]
		gmap.scatter(la_data+google_bus_la,lo_data+google_bus_lo,c=color[int(ii)-1], size=40, marker=False)
	gmap.draw('D:car/data/t66_bus_stop_and_roadtype_map_at_%s_%s.html'%(j,am_pm))
plt.clf()
plt.cla()



color_bar_ticklabels=['Clear Way','Lightly Jam','Start from Cross-Sections','Heavily Jam','Start From Bus Stops',
						'still','Others']
color=['lawngreen','pink','c','r','b','k','purple']
time=np.arange(14,18,1)

fig,ax=plt.subplots(2,2,figsize=(15,15))
ax=ax.ravel()
for i,j in enumerate (time):
	print (i,j)
	if j<10:
		time_str='2016-07-25 0'+str(j)	
	else:
		time_str='2016-07-25 '+str(j)

	if j<=12:
		am_pm='am'
	else:
		am_pm='pm'

	new_combine=t66_data[time_str]
	# col=np.array(color)[new_combine.label.values.astype(int)-1]
	ax[i].scatter(t66_stops.longitude+google_baidu_lo,t66_stops.latitude+google_baidu_la,c='gray',label='bus stop',s=100)
	for ii in np.unique(new_combine.label.values):
		ax[i].scatter(new_combine.longitude[new_combine.label==ii]+google_bus_lo,
			new_combine.latitude[new_combine.label==ii]+google_bus_la,
			label=color_bar_ticklabels[int(ii)-1],
			c=color[int(ii)-1],s=50)

	ax[i].set_ylabel('latitude',fontsize=15)
	ax[i].set_xlabel('longitude',fontsize=15)
	ax[i].set_title('%s %s'%(j,am_pm),fontsize=20)
	ax[i].set_xticks([])
	ax[i].set_yticks([])
	ax[i].legend(loc='best')
plt.savefig('D:car/data/t66_bus_stop_roadtype_14pm_17pm.png')
plt.show()



##T1
mean_la=np.mean(t1_stops.latitude.values)
mean_lo=np.mean(t1_stops.longitude.values)

color_bar_ticklabels=['Others','Still','Lightly Jam','Start from Cross-Sections','Heavily Jam','Start From Bus Stops',
						'Clear Way']
color=['purple','k','pink','c','r','b','lawngreen']

time=np.arange(7,19,1)
google_baidu_la=-0.0053289999999996951
google_baidu_lo=-0.0062369999999987158

google_bus_la=-0.0022963333333336777
google_bus_lo=0.0061813333333446963

for i,j in enumerate (time):
	print (i,j)
	if j<10:
		time_str='2016-05-24 0'+str(j)	
	else:
		time_str='2016-05-24 '+str(j)
	if j<=12:
		am_pm='am'
	else:
		am_pm='pm'

	new_combine=t1_data[time_str]
	gmap = gmplot.GoogleMapPlotter(mean_la,mean_lo,16)
	gmap.scatter(t1_stops.latitude.values+google_baidu_la, t1_stops.longitude.values+google_baidu_lo, 'gray', size=100, marker=False)
	for ii in np.unique(new_combine.label.values):
		la_data=new_combine['latitude'][new_combine.label==ii].values
		lo_data=new_combine['longitude'][new_combine.label==ii].values
		col=color[int(ii-1)]
		gmap.scatter(la_data+google_bus_la,lo_data+google_bus_lo,c=color[int(ii)-1], size=40, marker=False)
	gmap.draw('D:car/data/t1_bus_stop_and_roadtype_map_at_%s_%s.html'%(j,am_pm))


plt.clf()
plt.cla()
color_bar_ticklabels=['Others','Still','Lightly Jam','Start from Cross-Sections','Heavily Jam','Clear Way','Start From Bus Stops',
						]
color=['purple','k','pink','c','r','lawngreen','b']
time=np.arange(18,20,1)
time=np.arange(7,11,1)
time=np.arange(11,15,1)
time=np.arange(15,19,1)

fig,ax=plt.subplots(2,2,figsize=(15,15))
ax=ax.ravel()
for i,j in enumerate (time):

	if j<10:
		time_str='2016-05-24 0'+str(j)	
	else:
		time_str='2016-05-24 '+str(j)

	if j<=12:
		am_pm='am'
	else:
		am_pm='pm'

	new_combine=t1_data[time_str]
	# col=np.array(color)[new_combine.label.values.astype(int)-1]
	ax[i].scatter(t1_stops.longitude+google_baidu_lo,t1_stops.latitude+google_baidu_la,c='gray',label='bus stop',s=100)
	for ii in np.unique(new_combine.label.values):
		ax[i].scatter(new_combine.longitude[new_combine.label==ii]+google_bus_lo,
			new_combine.latitude[new_combine.label==ii]+google_bus_la,
			label=color_bar_ticklabels[int(ii)-1],
			c=color[int(ii)-1],s=50)

	ax[i].set_ylabel('latitude',fontsize=15)
	ax[i].set_xlabel('longitude',fontsize=15)
	ax[i].set_title('%s %s'%(j,am_pm),fontsize=20)
	ax[i].set_xticks([])
	ax[i].set_yticks([])
	ax[i].legend(loc='best')
plt.savefig('D:car/data/t1_bus_stop_roadtype_15pm_18pm.png')
plt.show()
