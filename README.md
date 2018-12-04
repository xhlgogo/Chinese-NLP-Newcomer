## -1. 爬取政府工作报告：
### 利用教育网的便利爬取全国所有省市的政府工作报告，使用selenium驱动chrome在静默状态下获得网页html,使用BeautifulSoup获取文本内容，累计获取13000余份政府工作报告；

## -2. 对爬取数据进行处理：
### 在政府网站获取历年各省市行政编码，使用gb2260第三方lib库传入地区名对应的行政编码后，获得省市区三级行政名，对爬取的文件进行“年省市区.txt”重命名，爬取的统计结果使用pyecharts绘制3D柱状图和地形图，代码文件为spider_result_plot.py；
![省级统计](https://github.com/xhlgogo/Chinese-NLP-Newcomer/blob/master/pyecharts_result/map.gif)
![年代统计](https://github.com/xhlgogo/Chinese-NLP-Newcomer/blob/master/pyecharts_result/3Dbar.gif)

## -3. 分词：
### 使用哈工大ltp的python封装：pyltp,分词任务为IO密集型，因此选择python多线程分词，代码文件为wordcut_pyltp.py；

## -4. lda：
### 使用gensim对537个省级政府工作报告进行lda文本主题分类，分类结果使用pyecharts绘制时间线散点图，代码文件为gensim_lad.py；
![lda主题关键词](https://github.com/xhlgogo/Chinese-NLP-Newcomer/blob/master/pyecharts_result/lda_timeline.gif)

## -5. k-means：
### 使用sklearn对537个省级政府工作报告进行k-means文本聚类，分类结果使用pyecharts二维散点图，代码文件为sklearn_kmeans.py.
![k-means聚类结果](https://github.com/xhlgogo/Chinese-NLP-Newcomer/blob/master/pyecharts_result/k-means%E8%81%9A%E7%B1%BB.png)
