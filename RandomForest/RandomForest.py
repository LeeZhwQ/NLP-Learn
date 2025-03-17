#构建随机森林回归模型

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing #加利福尼亚房价数据集
from sklearn import preprocessing


ca_house = fetch_california_housing()
ca_feature_name = ca_house.feature_names
ca_feature = ca_house.data
ca_target = ca_house.target

#print(boston_feature_name)

#print(boston_house.DESCR)

rf = RandomForestRegressor(n_estimators=15)
rf = rf.fit(ca_feature,ca_target)

answer1 = rf.predict(ca_feature)
print(answer1)

#决策树模型，来比较
from sklearn import tree
dst = tree.DecisionTreeRegressor()
dst.fit(ca_feature,ca_target)

print(dst.predict(ca_feature))
answer2 = dst.predict(ca_feature)

#来看看哪个得分高
from sklearn.metrics import mean_squared_error
meanerror1 = mean_squared_error(ca_target,answer1)
meanerror2 = mean_squared_error(ca_target,answer2)

print("score of forest: {0} , score of tree: {1} ".format(meanerror1,meanerror2))

'''事实表明其实决策树在该数据集下的表现要比随机森林好'''
