from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer  #文本特征向量化模块
from sklearn.naive_bayes import MultinomialNB #导入贝叶斯模型
from sklearn.metrics import classification_report

#数据获取
data = load_breast_cancer()
print(len(data.data))

#数据预处理：分割测试集与训练集，向量化
X_train,X_test,y_train,y_test = train_test_split(data.data,data.target,test_size=0.25,random_state=33)

'''
#向量化
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.fit_transform(X_test)
'''

#训练
model = MultinomialNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test) #预测

print("the Accuracy is " , model.score(X_test,y_test))
print(classification_report(y_test,y_pred,target_names=data.target_names))
