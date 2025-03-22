import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load data

dataset = loadtxt('pima-indians-diabetes.csv',delimiter=",")
#分割结果和数据
X = dataset[:,0:8]
Y = dataset[:,8]

seed = 8
test_size = 0.4
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size,random_state=seed)
#fit model
model = XGBClassifier()
model.fit(X_train,Y_train,eval_metric='error')

#predict
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
#evaluate
accuracy = accuracy_score(Y_test,predictions)
print("Accuracy: " , accuracy * 100.0 , "%")
