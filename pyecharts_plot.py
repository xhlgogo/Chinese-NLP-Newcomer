# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:43:50 2018

@author: xhlgogo
"""

# =============================================================================
# from pyltp import SentenceSplitter
# from pyltp import Segmentor
# segmentor = Segmentor()
# sents = segmentor.segment('元芳你怎么看？我就趴窗口上看呗！')
# print('\n'.join(sents))
# segmentor.release()
# =============================================================================

# =============================================================================
# import jieba
# import jieba.analyse
# 
# with open('2012重庆市万州区.txt','r',encoding="utf-8") as file:
#     file_lines = file.readlines()
# 
# file_togather = str(file_lines)
# test_list = jieba.analyse.extract_tags(file_togather, topK=20, withWeight=True, allowPOS=("a"))
# print(test_list)
# 
# =============================================================================
import os
from pyecharts import Map
from pyecharts import Bar3D
from pyecharts.engine import create_default_environment

forder_list = ["北京市","天津市","上海市","河北省","山西省","辽宁省","吉林省","重庆市"
                      ,"黑龙江","江苏省","浙江省","安徽省","福建省","江西省","山东省","河南省"
                      ,"湖北省","湖南省","广东省","海南省","四川省","贵州省","云南省","陕西省"
                      ,"甘肃省","青海省","内蒙古","广西省","西藏省","宁夏省","新疆省"]

# maptype='china' 只显示全国直辖市和省级
# 数据只能是省名和直辖市的名称
provice_list = []
for provice in forder_list:
    if "市" in provice:
        provice_list.append(provice.replace("市",''))
    elif "省" in provice:
        provice_list.append(provice.replace("省",''))
    else:
        provice_list.append(provice)
        
count_calue = {}
for forder in forder_list:
    file_list = os.listdir("E:/Program Files/workspace/report/"+forder)
    count_time = {}
    for file in file_list:
        if file[0:4] not in count_time.keys():
            count_time[file[0:4]] = 1
        else:
            count_time[file[0:4]] = 1 + count_time[file[0:4]]
    
    count_calue[forder[0:2]] = count_time

count_calue["内蒙古"] = count_calue.pop("内蒙")
count_calue["黑龙江"] = count_calue.pop("黑龙")
    
count_total = []
for provice in provice_list:
    count_total.append(sum(count_calue[provice].values()))
    
bar3d = Bar3D("政府报告年代统计", width=1200, height=600)
x_axis = provice_list
y_axis = [str(time) for time in range(1986,2019)]
data = []
for sheng in count_calue.keys():
    for year in count_calue[sheng]:
        temp = []
        temp.append(sheng)
        temp.append(year)
        temp.append(count_calue[sheng][year])
        data.append(temp)
        
range_color = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf',
               '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
            
map_test = Map("中国地图",'政府报告省级统计', width=1200, height=600)
map_test.add("", provice_list, count_total, visual_range=[0, 1020],
             maptype='china', is_visualmap=True,
             visual_text_color='#000',is_label_show=True, is_more_utils=True)
map_test.show_config()
map_test.render("政府报告省级统计.html")

               
bar3d.add("", x_axis, y_axis, data, is_visualmap=True,
    visual_range=[0, 120], visual_range_color=range_color,
    grid3d_width=400, grid3d_depth=120,grid3d_shading="realistic",
    xaxis_interval=0, xaxis_rotate=30,
    xaxis3d_name ="省份",yaxis3d_name ="时间",
    xaxis3d_interval=0,yaxis3d_interval=2,
    is_grid3d_rotate=True, is_more_utils=True)
bar3d.render("政府报告年代统计.html")

# =============================================================================
# env=create_default_environment("pdf")
# env.render_chart_to_file(map_test,path="政府报告省级统计.pdf")
# env.render_chart_to_file(bar3d,path="政府报告年代统计.pdf")
# =============================================================================

     