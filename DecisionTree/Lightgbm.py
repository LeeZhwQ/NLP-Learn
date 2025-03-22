import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

iris = load_iris()
data = iris.data
target = iris.target
X_train,X_test,Y_train,Y_test = train_test_split(data,target,test_size=0.2)

#创建lgb数据格式
lgb_train = lgb.Dataset(X_train,Y_train)
lgb_eval = lgb.Dataset(X_test,Y_test,reference=lgb_train)

params =   {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective' : 'regression',
    'metric' : {'l2','auc'},
    'num_leaves':31,
    'learning_rate':0.05,
    'feature_fraction':0.9,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'verbose' : 1
}

print("start training")

gbm = lgb.train(params,lgb_train,num_boost_round=20,valid_sets=lgb_eval,early_stopping_rounds=3)

gbm.save_model('model.txt')#保存模型
print("start working")

y_pred = gbm.predict(X_test,num_iteration=gbm.best_iteration)

print("the rmse of prediction is " , mean_squared_error(Y_test,y_pred)**0.5)#均方根误差
