import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.grid_search import GridSearchCV
from datetime import datetime

train_url="C:/Users/dell/Downloads/train.csv"
test_url="C:/Users/dell/Downloads/test.csv"
train=pd.read_csv(train_url)
test=pd.read_csv(test_url)

temp=pd.DatetimeIndex(train["datetime"])
train['year'] = temp.year
train['month'] = temp.month
train['hour'] = temp.hour
train['weekday'] = temp.weekday

temp = pd.DatetimeIndex(test['datetime'])
test['year'] = temp.year
test['month'] = temp.month
test['hour'] = temp.hour
test['weekday'] = temp.weekday

features = ['season', 'holiday', 'workingday', 'weather',
        'temp', 'atemp', 'humidity', 'windspeed', 'year',
         'month', 'weekday', 'hour']

for col in ['casual', 'registered', 'count']:
    train['log-'+col]=train["col"].apply(lambda x:np.log1p(x))

temp = pd.DatetimeIndex(train['datetime'])
training=train[temp.day<=16]
validation=train[temp.day>16]

param_grid = {'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [10, 15, 20],
              'min_samples_leaf': [3, 5, 10, 20],
              }

est=ensemble.GradientBoostingRegressor(n_estimators=500)
gs_cv=GridSearchCV(est,param_grid,n_jobs=4).fit(training["features"],training["log-count"])
gs_cs.best_params_

error_train=[]
error_validation=[]
for k in range(10, 501, 10):
    est=ensemble.GradientBoostingRegressor(n_estimators=k,learning_rate=0.05,max_depth=10,min_samples_leaf=20)
    est.fit(training[features],training["log-count"])
    result = clf.predict(training[features])
    error_train.append(
        mean_absolute_error(result, training['log-count']))
 
    result = clf.predict(validation[features])
    error_validation.append(
        mean_absolute_error(result, validation['log-count']))  


    def merge_predict(model1, model2, test_data):
    p1 = np.expm1(model1.predict(test_data))
    p2 = np.expm1(model2.predict(test_data))
    p_total = (p1+p2)
    return(p_total)
est_casual = ensemble.GradientBoostingRegressor(n_estimators=80, learning_rate = .05)
est_registered = ensemble.GradientBoostingRegressor(n_estimators=80, learning_rate = .05)
param_grid2 = {'max_depth': [10, 15, 20],
              'min_samples_leaf': [3, 5, 10, 20],
              }
 
gs_casual = GridSearchCV(est_casual, param_grid2, n_jobs=4).fit(training[features], training['log-casual'])
gs_registered = GridSearchCV(est_registered, param_grid2, n_jobs=4).fit(training[features], training['log-registered'])      
 
result3 = merge_predict(gs_casual, gs_registered, test[features])
df=pd.DataFrame({'datetime':test['datetime'], 'count':result3})
df.to_csv('results3.csv', index = False, columns=['datetime','count'])

est_casual = ensemble.GradientBoostingRegressor(
    n_estimators=80, learning_rate = .05, max_depth = 10,min_samples_leaf = 20)
est_registered = ensemble.GradientBoostingRegressor(
    n_estimators=80, learning_rate = .05, max_depth = 10,min_samples_leaf = 20)
 
est_casual.fit(train[features].values, train['log-casual'].values)
est_registered.fit(train[features].values, train['log-registered'].values)
result4 = merge_predict(est_casual, est_registered, test[features])
 
df=pd.DataFrame({'datetime':test['datetime'], 'count':result4})
df.to_csv('C:/Users/dell/Downloads/count.csv', index = False, columns=['datetime','count'])










