# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:52:20 2018

@author: xhlgogo
"""
import os
import json
import random
from pyecharts import Scatter
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


label_names = ["北京市","天津市","上海市","河北省","山西省","辽宁省","吉林省","重庆市"
               ,"黑龙江省","江苏省","浙江省","安徽省","福建省","江西省","山东省","河南省"
               ,"湖北省","湖南省","广东省","海南省","四川省","贵州省","云南省","陕西省"
               ,"甘肃省","青海省","内蒙古自治区","广西壮族自治区","西藏自治区","宁夏回族自治区","新疆维吾尔自治区"]
lable_dict = {}
with open("E:/Program Files/workspace/report_sheng/lable_dict.json", 'r', encoding="UTF-8") as json_file:
    label_dict = json.loads(json_file.read())

#用于获取词频
def get_count_vect(corpus):
    #将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    count_vector = CountVectorizer()
    #matrix词频的稀疏矩阵表示。括号中的第一个元素表示的是行数，第二个元素表示的是列数。
    matrix= count_vector.fit_transform(corpus)
    #对应列的特征词
    feature_names = count_vector.get_feature_names()
# =============================================================================
#     #toarray为完整矩阵表示，与matrix是一个东西
#     feature_array = matrix.toarray()
# =============================================================================
    return matrix,feature_names

#用于获取tf-idf
def get_tfidf(count_vect_matrix):
    
    feature_tf = TfidfTransformer(use_idf=False).fit_transform(count_vect_matrix)
    
    return feature_tf

#用于将样本分为训练集、测试集
def part_features(features,labels,part=(3/4)):
    
    #分省市统计各省市文件的下标
    label_cunt = []
    for label in range(31):
        temp_cunt = [index for index in range(len(labels)) if labels[index] == label]
        label_cunt.append(temp_cunt)
    
    #按比例随机抽取各省市训练集下标,并将下标从小到大排序
    train_list = []
    for label in label_cunt:
        temp_cunt = random.sample(range(len(label)),int(len(label)*part))
        temp_train = [label[index] for index in temp_cunt]
        train_list.extend(temp_train)
    train_list.sort()
    
    #测试集样本下标
    test_list = [index for index in range(len(labels)) if index not in train_list]
    
    #按下标抽取训练样本、样本标签
    train_data = [features[i] for i in train_list]
    train_label = [labels[i] for i in train_list]
    
    #按下标抽取测试样本、样本标签
    test_data = [features[i] for i in test_list]
    test_label = [labels[i] for i in test_list]
    
    return train_data,train_label,test_data,test_label

#用于读取相似获取
def get_feature(path):

    words = []
    labels = []
    file_list = os.listdir(path)
    for file_name in file_list:
        value_str = ''
        (file_fir,extension) = os.path.splitext(file_name)
        with open(path+file_name, 'r', encoding="UTF-8") as json_file:
                temp_dict = json.loads(json_file.read())
        for value in temp_dict.values():
            for word in value:
                value_str = value_str + word.replace("\n",' ')
        words.append(value_str)
        labels.append(label_dict[file_fir[4:]])
    
    return words,labels

#读取特征，将地名、人名、机构名替换成词性
def get_feature_1(path):

    not_keep = ["nh","ns","ni"]
    
    words = []
    labels = []
    file_list = os.listdir(path)
    for file_name in file_list:
        value_str = ''
        (file_fir,extension) = os.path.splitext(file_name)
        with open(path+file_name, 'r', encoding="UTF-8") as json_file:
                temp_dict = json.loads(json_file.read())
        for key,value in temp_dict.items():
            key_split = key.split()
            find = False
            for index in range(len(key_split)):
                #若存在地名、人名、机构名，则将其替换为词性
                if key_split[index] in not_keep:
                    find = True
                    for item in value:
                        value_split = item.split()
                        value_split[index] = key_split[index]
                        value_str = value_str + ' '.join(value_split) + ' '
            #若不存在地名、人名、机构名
            if not find:
                for item in value:
                    value_str = value_str + item.replace("\n",' ')
        words.append(value_str)
        labels.append(label_dict[file_fir[4:]])
    
    return words,labels

#读取特征，将地名、人名、机构名去掉
def get_feature_2(path):

    not_keep = ["nh","ns","ni"]
    
    words = []
    labels = []
    full_names = []
    file_list = os.listdir(path)
    for file_name in file_list:
        value_str = ''
        (file_fir,extension) = os.path.splitext(file_name)
        with open(path+file_name, 'r', encoding="UTF-8") as json_file:
                temp_dict = json.loads(json_file.read())
        for key,value in temp_dict.items():
            key_split = key.split()
            find = False
            for index in range(len(key_split)):
                #若存在地名、人名、机构名，则将其赋空
                if key_split[index] in not_keep:
                    find = True
                    for item in value:
                        value_split = item.split()
                        #使用del或pop会影响原有index结构，直接赋空
                        value_split[index] = ''
                        value_str = value_str + ' '.join(value_split) + ' '
            #若不存在地名、人名、机构名
            if not find:
                for item in value:
                    value_str = value_str + item.replace("\n",' ')
        words.append(value_str)
        if "省" in file_fir:
            index = file_fir.index("省")
        elif "区" in file_fir:
            index = file_fir.index("区")
        labels.append(label_dict[file_fir[4:(index+1)]])
        full_names.append(file_fir)
    
    return words,labels,full_names


#使用卡方分布选区特征
def select_feature(features,labels,num=93):
     print("start select features: ",num)
      #使用卡方分布选区10个特征
     model1 = SelectKBest(chi2, k=num)#选择k个最佳特征
     new_features = model1.fit_transform(features, labels)#iris.data是特征数据，iris.target是标签数据，该函数可以选择出k个特征
     return new_features
 
def read_json(path_name):
    with open(path_name, 'r', encoding="UTF-8") as json_file:
        read_dict = json.loads(json_file.read())
    return read_dict

def write_json(path_name,file_dict):
    with open(path_name, 'w', encoding="UTF-8") as json_file:
        json.dump(file_dict,json_file)
        
        
def tsen_plot(name,weight,label):
    print("T-SNE start")
    # 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
    tsne = TSNE(n_components=2)
    decomposition_data = tsne.fit_transform(weight)
    
    x = []
    y = []
    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])
    
    scatter = Scatter(title=name,title_top='bottom',title_pos='center')
    for i in range(len(x)):
        scatter.add(str(label[i]),[x[i]],[y[i]],symbol_size=5,is_datazoom_show=True)
    scatter.render(name+".html")


# =============================================================================
# def move_file():
#     forder_list = ["北京市","天津市","上海市","河北省","山西省","辽宁省","吉林省","重庆市"
#                    ,"黑龙江","江苏省","浙江省","安徽省","福建省","江西省","山东省","河南省"
#                    ,"湖北省","湖南省","广东省","海南省","四川省","贵州省","云南省","陕西省"
#                    ,"甘肃省","青海省","内蒙古","广西省","西藏省","宁夏省","新疆省"]
#     
#     read_path = "E:/Program Files/workspace/report_similarity/"
#     write_path = "E:/Program Files/workspace/report_sheng/report_simi/"
#     forder = "内蒙古"
#     file_list = os.listdir(read_path+forder)
#     for file_name in file_list:
#         (file_fir,extension) = os.path.splitext(file_name)
#         if file_fir[-1] == "市":
#             print(file_fir)
#         for item in ["嘉峪关市"]:
#             if item == file_fir[-4:]:
#                 os.rename(read_path+forder+'/'+file_name, write_path+file_name)
#                 print(file_fir)
# =============================================================================
# =============================================================================
# if __name__=="__main__":
#     
#     path = "E:/Program Files/workspace/report_sheng/report_simi/"
# 
#     not_keep = ["nh","ns","ni"]
#     
#     words = []
#     labels = []
#     full_names = []
#     file_list = os.listdir(path)
#     for file_name in file_list:
#         value_str = ''
#         (file_fir,extension) = os.path.splitext(file_name)
#         with open(path+file_name, 'r', encoding="UTF-8") as json_file:
#                 temp_dict = json.loads(json_file.read())
#         for key,value in temp_dict.items():
#             key_split = key.split()
#             find = False
#             for index in range(len(key_split)):
#                 #若存在地名、人名、机构名，则将其赋空
#                 if key_split[index] in not_keep:
#                     find = True
#                     for item in value:
#                         value_split = item.split()
#                         #使用del或pop会影响原有index结构，直接赋空
#                         value_split[index] = ''
#                         value_str = value_str + ' '.join(value_split) + ' '
#             #若不存在地名、人名、机构名
#             if not find:
#                 for item in value:
#                     value_str = value_str + item.replace("\n",' ')
#         words.append(value_str)
#         if "省" in file_fir:
#             index = file_fir.index("省")
#         elif "区" in file_fir:
#             index = file_fir.index("区")
#         labels.append(label_dict[file_fir[4:(index+1)]])
#         full_names.append(file_fir)
# =============================================================================
