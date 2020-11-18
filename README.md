# digix2020_ctr_rank1
华为digix算法大赛2020机器学习赛道-ctr预估初赛/决赛rank1

队名：忑廊埔大战拗芭码（决赛更名为风犹惊入萧独夜）

分数：初赛A榜0.820305/B榜0.822074，决赛A榜0.822590/B榜0.814384

排名：初赛A榜rank1/B榜rank1，决赛A榜rank2/B榜rank1

项目的blog分享[敬请期待](https://blog.csdn.net/weixin_40174982/article/details/108880726)

南京之旅圆满结束，首冠到手，感谢队友！

## 项目环境

Python 3.8

lightgbm

gensim

sklearn

pandas

numpy

tqdm

networkx

## 处理流程

在ctr下创建data文件夹，并将训练集、测试集A、测试集B的csv文件放在ctr/data/

运行reduce/reduce.py进行数据压缩

运行full.py进行全特征模型的训练和推理，决赛B榜分数813

运行win.py进行滑窗模型的训练和推理，决赛B榜分数811

运行nounique.py进行部分特征模型的训练和推理，决赛B榜分数811

运行fusion.py得到三个模型结果的融合，决赛B榜分数814

result文件夹中可得到最终结果文件submission_f.csv
