# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:56:31 2018

@author: xhlgogo
"""
import tools
import numpy as np
from tqdm import trange
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

def naive_bayes_Multinomial(new_feature,labels_list):
    
    #随机将样本分为训练集、测试集
    train_data,train_label,test_data,test_label = tools.part_features(new_feature,labels_list)
    
    mnb = MultinomialNB(alpha=0.0001,fit_prior=True)
    mnb.fit(train_data,train_label)
#    predict_label = mnb.predict(test_data)
    
    print('Traing score : %.2f' % mnb.score(train_data,train_label))
    print('Testing score : %.2f' % mnb.score(test_data,test_label))
    
    return mnb.score(test_data,test_label)

def naive_bayes_Gaussian(new_feature,labels_list):
    
    #随机将样本分为训练集、测试集
    train_data,train_label,test_data,test_label = tools.part_features(new_feature,labels_list)
    
    gnb = GaussianNB()
    gnb.fit(train_data,train_label)
    
    print('Traing score : %.2f' % gnb.score(train_data,train_label))
    print('Testing score : %.2f' % gnb.score(test_data,test_label))
    
    return gnb.score(test_data,test_label)


def bayes_select_feature(model_name,word_matrix,labels_list,file_name):
    
    if model_name == "Gaussian":
        model=naive_bayes_Gaussian
    elif model_name == "Multinomial":
        model=naive_bayes_Multinomial
        
    score_dict = {}
    for num in trange(300,900):
        #使用卡方分布从词频特征抽取特征
        new_feature = tools.select_feature(word_matrix.toarray(),labels_list,num)
        
        #贝叶斯分类
        score = model(new_feature,labels_list)
        
        #记录卡方分布抽取数对应的分类精度
        score_dict[str(num)+ " features"] = score
        
        if score > 0.9:
            break
    
    max_item = max(score_dict.items(), key = lambda x: x[1])
    print("The best is ",max_item)
    
    score_dict["Best"] = [max_item[0], max_item[1]]
    tools.write_json("E:/Program Files/workspace/report_sheng/"+file_name+".json",score_dict)

def Multinomial_select_alpha(word_matrix,labels_list,feature_num):
    
    new_feature = tools.select_feature(word_matrix.toarray(),labels_list,feature_num)
    
    #随机将样本分为训练集、测试集
    train_data,train_label,test_data,test_label = tools.part_features(new_feature,labels_list)
    
    score_dict = {}
    for al in np.arange(0.0,0.12,0.0001):
        
        mnb = MultinomialNB(alpha=al,fit_prior=True)
        mnb.fit(train_data,train_label)
        
        print('Traing score : %.2f' % mnb.score(train_data,train_label))
        print('Testing score : %.2f' % mnb.score(test_data,test_label))
        
        score_dict["alpha = "+str(al)] = mnb.score(test_data,test_label)
    
    max_item = max(score_dict.items(), key = lambda x: x[1])
    print("The best is ",max_item)
    
    score_dict["Best"] = [max_item[0], max_item[1]]
    tools.write_json("E:/Program Files/workspace/report_sheng/Multinomial_alpha_score_dict.json",score_dict)


if __name__=="__main__":
    
    simi_path = "E:/Program Files/workspace/report_sheng/report_simi/"
    similarity_path = "E:/Program Files/workspace/report_sheng/report_similarity/"
    words,labels,full_names = tools.get_feature_2(similarity_path)
        
    word_matrix,word_names = tools.get_count_vect(words)
# =============================================================================
#     #市级最佳特征数，经卡方分布选取为897个
#     bayes_select_feature("Multinomial",word_matrix, labels, "Multinomial_score_dict-市级")
# =============================================================================
