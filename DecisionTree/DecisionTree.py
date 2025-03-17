import pandas as pd
from sklearn import preprocessing
from sklearn import tree

adult_data = pd.read_csv("DecisionTree.csv")

'''
print(adult_data.head())
print(adult_data.info())
print(adult_data.shape)
print(adult_data.columns)
'''

#区分特征值和目标值
features_columns = ['workclass' , 'education' , 'marital-status' , 'occupation' ,
                    'relationship' , 'race' , 'gender' , 'native-country' ]
label_columns = ['income']

features = adult_data[features_columns]
label = adult_data[label_columns]

#print(label.head(2))

features = pd.get_dummies(features)
#print(features.head(2))

#创建一个决策树分类器
clf = tree.DecisionTreeClassifier(criterion='entropy' , max_depth=4)
#拟合数据，进行训练
clf.fit(features.values , label.values)

print(clf.predict(features.values))

#进行可视化：

import pydotplus
from IPython.display import Image , display
from PIL import Image

dot_data = tree.export_graphviz(clf,
                                out_file=None,
                                feature_names= features.columns,
                                class_names= ['<=50K' , '>50K'],
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

img_path = 'decisionTree.png'
graph.write_png(img_path)

img = Image.open(img_path)
img.show()

