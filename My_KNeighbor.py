# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:32:58 2018

@author: xhlgogo
"""

import tools
from tqdm import trange
from tools import label_names,lable_dict
from sklearn.neighbors import KNeighborsClassifier

def KNeighbor(features,labels,neighbors):
    
    #随机将样本分为训练集、测试集
    train_data,train_label,test_data,test_label = tools.part_features(features,labels)
        
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(train_data, train_label)
    
#    predict_labe = knn.predict(test_data)
    score = knn.score(test_data,test_label)
    
    print("精度为： ",score)
#    tools.tsen_plot("KNeighbor",features,labels)
    return score

if __name__=="__main__":
    
    simi_path = "E:/Program Files/workspace/report_sheng/report_similarity/"
    features,labels,words,feature_names = tools.get_feature(simi_path)
    
    labels_list = []
    for i in range(len(labels)):
        labels_list.append(lable_dict[labels[i]])
    
    word_matrix,word_names = tools.get_count_vect(words)
    
    score_dict = {}
    for num in trange(10,1137):
        #使用卡方分布从词频特征抽取特征
        new_feature = tools.select_feature(word_matrix.toarray(),labels_list,num)
        
        for neibhor in range(3,32):
            #多项式贝叶斯分类
            score = KNeighbor(new_feature,labels_list,neibhor)
            #记录卡方分布抽取数对应的分类精度
            score_dict[str(num)+ " features and "+str(neibhor)+" neighbors"] = score
            
    max_item = max(score_dict.items(), key = lambda x: x[1])
    print("The best is ",max_item)
    
    score_dict["Best"] = [max_item[0], max_item[1]]
    tools.write_json("E:/Program Files/workspace/report_sheng/KNeighbor_score_dict.json",score_dict)
