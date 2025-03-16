from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request


#参数
args = Namespace(
    seed = 1234,
    data_file = "titanic.csv",
    train_size = 0.75,
    test_size = 0.25,
    num_epochs = 100,
)

np.random.seed(args.seed)


df = pd.read_csv(args.data_file , header = 0)
print(df.head())

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess(df):

    df = df.dropna()

    #删除基于文本的特征
    features_to_drop = ["name" , "cabin" , "ticket"]
    df = df.drop(features_to_drop , axis = 1)

    #将类别变量转换为数字变量
    categorical_features = ["sex","pclass","embarked"]
    df = pd.get_dummies(df,columns=categorical_features)

    return df

df = preprocess(df)
print(df.head())

#划分数据到训练集和测试集
mask = np.random.rand(len(df)) < args.train_size
train_df = df[mask]
test_df = df[~mask]
print("Train_size : {0} , test_size : {1}".format(len(train_df),len(test_df)))

#分离x与y
x_train = train_df.drop(["survived"],axis= 1)
y_train = train_df["survived"]
x_test = test_df.drop(["survived"],axis=1)
y_test = test_df["survived"]

#标准化训练数据 （mean = 0 ，std = 1）避免数值不稳定，拟合
x_scaler = StandardScaler()
x_scaler.fit(x_train)

#标准化x
standard_x_train = x_scaler.transform(x_train)
standard_x_test = x_scaler.transform(x_test)

#check
print("mean : " , np.mean(standard_x_train,axis=0))
print("std : " , np.std(standard_x_train , axis=0))

#初始化模型
log_reg = SGDClassifier(loss="log" , penalty="none",max_iter=args.num_epochs , random_state= args.seed)

#训练
log_reg.fit(X=standard_x_train,y=y_train)

#概率
pred_test = log_reg.predict_proba(standard_x_test)
print(pred_test)

#预测(未标准化
pred_train = log_reg.predict(x_train)
pred_test2 = log_reg.predict(x_test)
print(pred_train)

##评估指标
from sklearn.metrics import accuracy_score

#正确率
train_acc = accuracy_score(y_train , pred_train)
test_acc = accuracy_score(y_test , pred_test2)
print("train_acc : {0:.2f} , test_acc : {1:.2f}".format(train_acc,test_acc))

#k折验证概率
from sklearn.model_selection import cross_val_score
scores = cross_val_score(log_reg,standard_x_train,y_train,cv=10,scoring="accuracy")
print("scores: " , scores)
print("mean : " , scores.mean())
print("standard deviation : " ,scores.std())
