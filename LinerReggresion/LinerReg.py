import os
import numpy as np
import pandas as pd


np.random.seed(36)

import matplotlib
import seaborn
import matplotlib.pyplot as plot

from sklearn import datasets

#读取数据

housing = pd.read_csv('kc_train.csv')
target = pd.read_csv('kc_train_2.csv')
t = pd.read_csv('kc_test.csv')

#预处理
housing.info()

# 特征缩放
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(housing)  # 内部拟合
minmax_housing = minmax_scaler.transform(housing)
scalar_housing = pd.DataFrame(minmax_housing,columns=housing.columns)

mm = MinMaxScaler()
mm.fit(t)
scalar_t = mm.transform(t)
scalar_t = pd.DataFrame(scalar_t,columns=t.columns)

#选择基于梯度下降的模型
from sklearn.linear_model import LinearRegression
LR_re = LinearRegression()
LR_re.fit(scalar_housing,target)

#预测
from sklearn.metrics import mean_squared_error
preds = LR_re.predict(scalar_housing)
mse = mean_squared_error(preds,target)

print(mse)

#绘图进行比较
plot.figure(figsize=(10,7))
num = 100
x = np.arange(1,num + 1) # 取一百个点
plot.plot(x,target[:num],label = 'target')
plot.plot(x,preds[:num], label = 'predict')
plot.legend(loc = 'upper right')

plot.show()

#输出测试数据test

result = LR_re.predict(scalar_t)
df_result = pd.DataFrame(result)
df_result.to_csv("result.csv")
