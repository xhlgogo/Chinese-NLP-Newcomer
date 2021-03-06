# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:31:05 2018

@author: xhlgogo
"""
import os
import json
from pyecharts import EffectScatter, Page, Timeline
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from pyltp import Segmentor
from pyltp import Postagger
import pandas as pd


read_path = "E:/Program Files/workspace/report_sheng/report_sentence/"
write_path = "E:/Program Files/workspace/report_sheng/report_word/"
write_path_cut = "E:/Program Files/workspace/report_sheng/report_vec/"
write_path_cut_txt = "E:/Program Files/workspace/report_sheng/report_cut_txt/"


def get_content(file_name):    

    with open(read_path+file_name,'r',encoding="utf-8") as file:
        content = file.readlines()
        
    keep = ['a', 'b', 'd', 'i', 'j', 'n', 'nh',
            'ni', 'nl', 'ns', 'nt', 'nz', 'v']
    #对不含数字的句子分词，只保留特定词性的词
    words_postags = []
    for element in content:
        words = list(segmentor.segment(element))
        tags = postagger.postag(words)
        for i in range(len(words)-1, -1, -1):
            if tags[i] not in keep:
                del words[i]
        for word in words:
            if word not in stop_words and len(word)>1:
                words_postags.append(word)
    
    #写分词结果,存为json文件      
    with open(write_path_cut+file_name[:-4]+".json", 'w') as json_file:
            json.dump(words_postags, json_file)
    
    #写分词结果,存为txt文件    
    with open(write_path_cut_txt+file_name[:-4]+".txt", 'w') as txt_file:
            txt_file.write(' '.join(words_postags))
    
    return words_postags



if __name__=="__main__":


    file_list = os.listdir(read_path)

    stop_words = []
    with open('D:/Program Files/workspace/stop_words.txt', "r", encoding="UTF-8") as fStopWords:
        for element in fStopWords.readlines():
            stop_words.append(element.strip())
    
    
    LTP_DATA_DIR = 'E:/Program Files/workspace/ltp_data_v3.4.0'  # ltp模型目录的路径
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    segmentor = Segmentor()  # 初始化实例
    segmentor.load_with_lexicon(cws_model_path, LTP_DATA_DIR+'/user_dict.txt') # 加载模型，第二个参数是您的外部词典文件路径
    postagger = Postagger() # 初始化实例
    postagger.load(pos_model_path)  # 加载模型
    
    combain_comtent = []
    for file in file_list:
        combain_comtent.append(get_content(file))
        
    segmentor.release()  # 释放模型
    
    dictionary = Dictionary(combain_comtent)
    corpus = [ dictionary.doc2bow(text) for text in combain_comtent]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=31)
    
    #词典的保存
    dictionary.save_as_text(write_path+"dictionary.txt")
    # lda模型保存
    lda.save(write_path+"model")
    
    for file in lda.print_topics(31):
        print(file[0])
    
    
    topic_list = []
    for i in lda.get_document_topics(corpus):
        listj=[]
        for j in i:
            listj.append(j[1])
        topic_list.append(listj.index(max(listj)))
        
    file_dict = {}
    for file in file_list:
        file_dict[file[:-4]] = topic_list[file_list.index(file)]
    
    #汇总结果
    temp_list = []
    for key,value in file_dict.items():
        temp_list.append([key,value])
    file_result = pd.DataFrame(temp_list, columns=("文件名","主题"))
    file_result.to_excel(write_path+"file_result.xlsx")
    
    
    forder_list = ["北京市","天津市","上海市","河北省","山西省","辽宁省","吉林省","重庆市"
                       ,"黑龙江","江苏省","浙江省","安徽省","福建省","江西省","山东省","河南省"
                       ,"湖北省","湖南省","广东省","海南省","四川省","贵州省","云南省","陕西省"
                       ,"甘肃省","青海省","内蒙古自治区","广西壮族自治区","西藏自治区","宁夏回族自治区","新疆维吾尔自治区"]
     
    #各省主题统计结果                 
    for forder in forder_list:
        temp_dict = {}
        for key,value in file_dict.items():
            if key[4:] == forder :
                if str(value) not in temp_dict.keys():
                    temp_dict[str(value)] = 1
                else:
                    temp_dict[str(value)] = temp_dict[str(value)] + 1
        temp_list = []
        for key,value in temp_dict.items():
            temp_list.append([key,value])
        pd.DataFrame(temp_list, columns=("主题","次数")).to_excel(write_path+forder+".xlsx")



# =============================================================================
# combain_comtent = []
# for file in file_list:
#     with open(write_path_cut+file[:-4]+".json", 'r') as json_file:
#         combain_comtent.append(json.loads(json_file.read()))
# =============================================================================


#绘制文档主题分布
# =============================================================================
# topic_word = []
# topic_word_gailv = []
# for topic_id in range(num_topics):
#     print('第%d个主题的词与概率如下：\t' % topic_id)
#     term_distribute_all = lda.get_topic_terms(topicid=topic_id)
#     term_distribute = term_distribute_all[:num_show_term]
#     term_distribute = np.array(term_distribute)
#     term_id = term_distribute[:, 0].astype(np.int)
#     print('词：\t', end='  ')
#     temp_word = []
#     for t in term_id:
#         temp_word.append(dictionary.id2token[t])
#         print(dictionary.id2token[t], end=' ')
#     topic_word.append(temp_word)
#     print('\n概率：\t', term_distribute[:, 1])
#     topic_word_gailv.append(term_distribute[:, 1])
# 
# page = Page(page_title="31个主题分类")
# timeline = Timeline(is_auto_play=True, is_loop_play=True, timeline_bottom=0)
# 
# x_axis = [0,1,2,3,4,5,6]
# for k in range(31):
#     es = EffectScatter("主题"+str(k))
#     #生成各散点图
#     for i in range(7):
#         es.add(topic_word[k][i], [x_axis[i]], [topic_word_gailv[k][i]],
#                effect_period=3, symbol='pin')
#     #添加散点图到page
#     page.add(es)
#     #添加散点图到timeline
#     timeline.add(es, "主题"+str(k))
# 
# page.render("Page.html")
# ===========================================================
# timeline.render("Timeline.html")==================


"""
# 构建训练语料
dictionary = Dictionary(train_content)
corpus = [ dictionary.doc2bow(text) for text in train_content]

#tf-idf模型
corpus_tfidf  = models.TfidfModel(corpus)[corpus]
#保存tf-idf模型，读取模型
#tfidf.save(write_path+"model.tfidf")
#tfidf = models.TfidfModel.load(write_path+"model.tfidf")

num_topics = 31     #主题数
num_show_term = 7   # 每个主题下显示几个词

# lda模型训练
lda = LdaModel(corpus, id2word=dictionary, num_topics=num_topics)#, minimum_probability=0.001)
#lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=31)


# lda模型保存，读取
lda.save(write_path+"model")
# Load a potentially pretrained model from disk.
"""

