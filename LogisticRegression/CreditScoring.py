import pandas as pd

from LogisticReg import x_scaler

data = pd.read_csv("cs-training.csv")
print(data.head())
print(data.shape)

data = data.dropna()
y = data['SeriousDlqin2yrs']
x = data.drop('SeriousDlqin2yrs',axis=1)


from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2)


lr = LogisticRegression(multi_class='ovr' , solver='sag',class_weight='balanced')
scaler = StandardScaler()

x_scaler = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

lr.fit(x_scaler,y_train)
score = lr.score(x_scaler,y_train)
print(score)

from sklearn.metrics import accuracy_score
train_score = accuracy_score(y_train,lr.predict(x_train))
test_score = lr.score(x_test,y_test)
print("训练集准确率： " , train_score)
print("测试集准确率： ", test_score)

#召回率测算
from sklearn.metrics import recall_score
train_recall = recall_score(y_train,lr.predict(x_scaler),average='macro')
test_recall = recall_score(y_test,lr.predict(x_test),average='macro')
print("训练集召回率： " , train_recall)
print("测试集召回率： " , test_recall)

#提高阈值,到0.3
import numpy as np
y_pro = lr.predict_proba(x_test)
y_pred2 = [list(p >= 0.3).index(1) for i,p in enumerate(y_pro)]
train_score_new = accuracy_score(y_test,y_pred2)
print(train_score_new)
