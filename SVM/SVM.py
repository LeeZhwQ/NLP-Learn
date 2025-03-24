import sys
import os
import jieba
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


news_file = 'svm_data/cnews.train.txt'
test_file = "svm_data/cnews.test.txt"
output_word_file = "cnews_dict.txt"
output_word_test_file = "cnews_dict_test.txt"
feature_test_file = "cnews_feature_test_file.txt"
feature_file = "cnews_feature_file.txt"
model_file_name = 'cnews_model'
with open(news_file,'r',encoding='utf-8') as f:
    lines = f.readlines()

label , content = lines[0].strip('\r\n').split('\t')
print(content)

words_iter = jieba.cut(content)
print('/'.join(words_iter))

#定义分词函数，写入相关文件
def generate_word_file(input_char_file,output_file):
    with open(input_char_file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    with open(output_file,'w',encoding='utf-8') as f:
        for line in lines:
            label,content = line.strip('\r\n').split('\t')
            words_iter = jieba.cut(content)
            word_content = ''
            for word in words_iter:
                word = word.strip(' ')
                if word != '':
                    word_content += word + ' '
            out_line = '%s\t%s\n' % (label,word_content.strip(' '))
            f.write(out_line)

generate_word_file(news_file,output_word_file)
generate_word_file(test_file,output_word_test_file)
print("分词完成")

#对新闻的分类生成
class Category:
    def __init__(self,category_file):
        self._category_to_id = {}
        with open(category_file,'r',encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            category , idx = line.strip('\r\n').split('\t')
            idx = int(idx)
            self._category_to_id[category] = idx

    def category_to_id(self,category):
        return self._category_to_id[category]

    def size(self):
        return len(self._category_to_id)

category_file = 'svm_data/cnews.category.txt'
category_vocab = Category(category_file)
print(category_vocab.size())

#进行词频统计并过滤，生成词id，dict

def generate_feature_dict(train_file,feature_threshold = 10):
    feature_dict = {}
    with open(train_file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        label, content = line.strip('\r\n').split('\t')
        for word in content.split(' '):
            if not word in feature_dict:
                feature_dict.setdefault(word,0)
            feature_dict[word] += 1
    filter_feature_dict = {}
    for feature_name in feature_dict:
        if feature_dict[feature_name] < feature_threshold:
            continue
        if not feature_name in filter_feature_dict:
            filter_feature_dict[feature_name] = len(filter_feature_dict) + 1
    return filter_feature_dict

feature_dict = generate_feature_dict(output_word_file,feature_threshold=200)
print(len(feature_dict))

#对每一篇新闻构造稀疏词向量
def generate_feature_line(line,feature_dict,category_vocab):
    label,content = line.strip('\r\n').split('\t',1)
    label_id = category_vocab.category_to_id(label)
    feature_example = {}
    for word in content.split(' '):
        word = word.strip()
        if not word:
            continue
        if not word in feature_dict:
            continue
        feature_id = feature_dict[word]
        feature_example.setdefault(feature_id,0)
        feature_example[feature_id] += 1
    sorted_features = sorted(feature_example.items(),key= lambda d:d[0])
    feature_parts = [f"{fid}:{count}" for fid, count in sorted_features]
    # 使用制表符分隔标签和特征，特征间用空格分隔
    feature_line = f"{label_id}\t{' '.join(feature_parts)}"
    return feature_line

#循环每一篇文章，得到词向量化的文件

def convert_raw_to_feature(raw_file,feature_file,feature_dict,category_vocab):
    with open(raw_file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    with open(feature_file,'w',encoding='utf-8') as f:
        for line in lines:
            feature_line = generate_feature_line(line,feature_dict,category_vocab)
            f.write('%s\n' % feature_line)

convert_raw_to_feature(output_word_file,feature_file,feature_dict,category_vocab)
convert_raw_to_feature(output_word_test_file,feature_test_file,feature_dict, category_vocab)

print("构造词向量完成")

#生成训练数据
def load_data(feature_file):
    with open(feature_file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    labels = []
    features = []
    for line in lines:
        parts = line.strip('\r\n').split('\t',1)
        label = int(parts[0])
        labels.append(label)
        feature = {}
        if len(parts) > 1:  # 处理存在特征的情况
            # 关键修改：先按空格分割特征项
            for item in parts[1].split(' '):  # 用空格分割每个特征项
                if ':' in item:
                    fid, val = item.split(':', 1)  # 只分割一次防止意外
                    feature[int(fid)] = int(val)
        features.append(feature)
    return labels,features

train_label,train_value = load_data(feature_file)
train_test_label,train_test_value = load_data(feature_test_file)

#使用向量化特征,可能是这个的问题
vectorizer = TfidfVectorizer(analyzer= lambda x:x)
X_train = vectorizer.fit_transform([' '.join(str(k) for k in feature.keys()) for feature in train_value])
X_test = vectorizer.fit_transform([' '.join(str(k) for k in feature.keys()) for feature in train_test_value])
y_train = train_label
y_test = train_test_label

#训练模型
model = svm.SVC(kernel='rbf' , C=5, gamma=0.5,probability=True)
model.fit(X_train,y_train)

#预测，给出准确值
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
