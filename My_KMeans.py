# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 19:08:35 2018

@author: xhlgogo
"""
import tools
import numpy as np
from tqdm import trange
import multiprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

if __name__=="__main__":
    
    simi_path = "E:/Program Files/workspace/report_sheng/report_simi/"
    words,labels,full_names = tools.get_feature_2(simi_path)

    word_matrix,word_names = tools.get_count_vect(words)
    
    #使用卡方分布从词频特征抽取特征
    new_feature = tools.select_feature(word_matrix.toarray(),labels,897)
    
    print('start training')
    
    SSE = []    #用于存放每个训练模型的inertia_
    SSE_d1 = [] #sse的一阶导数
    SSE_d2 = [] #sse的二阶导数
    models = [] #保存每次的模型
    for num in trange(31, 310):
        #指定簇数训练模型，指定初始值选择算法为k-means++，指定进程数为cpu数
        kmeans_model = MiniBatchKMeans(n_clusters=num, init='k-means++', init_size=3*num)
# =============================================================================
#         kmeans_model = KMeans(n_clusters=num, init='k-means++',
#                               n_jobs=multiprocessing.cpu_count())
# =============================================================================
        kmeans_model.fit(new_feature)
        SSE.append(kmeans_model.inertia_)  # 保存每一个k值的SSE值
        print('{} Means SSE loss = {}'.format(num, kmeans_model.inertia_))
        models.append(kmeans_model)
    # 求一阶导数
    SSE_d1 = [((SSE[i-1]-SSE[i])/2) for i in range(1, len(SSE))]
    # 求二阶导数
    SSE_d2 = [((SSE_d1[i-1]-SSE_d1[i])/2) for i in range(1, len(SSE_d1))]
    
    best_model = models[SSE_d2.index(max(SSE_d2)) + 1]
    print("SSE_d2.index(max(SSE_d2)) is \n",SSE_d2.index(max(SSE_d2)))
    
    new_labels = best_model.labels_
    
    import pandas as pd
    result = pd.DataFrame()
    result["full_names"] = full_names
    result["yuan_labels"] = labels
    result["KMeans_labels"] = new_labels
    result.to_excel("E:/Program Files/workspace/report_sheng/KMeans_result.xlsx")
    
    
# =============================================================================
#     SSE_length = len(SSE)
#     for i in range(1, SSE_length):
#         SSE_d1.append((SSE[i - 1] - SSE[i]) / 2)
#     for i in range(1, len(SSE_d1) - 1):
#         SSE_d2.append((SSE_d1[i - 1] - SSE_d1[i]) / 2)
# =============================================================================

    
    
    
    
    
    

# =============================================================================
#     tfidf = tools.get_tfidf(word_matrix.toarray())
#     
#     #将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
#     weight = tfidf.toarray()
#     
#     print('Start Kmeans:')
#     clf = KMeans(n_clusters=31)
#     clf.fit(weight)
# # =============================================================================
# #     #31个中心点
# #     print(clf.cluster_centers_)
# # =============================================================================
#     #用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
#     print("inertia_ is ", clf.inertia_)
#     tools.tsen_plot("Kmeans",weight,clf.labels_)
# =============================================================================


