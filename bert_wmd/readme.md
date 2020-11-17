# bert+wmd模型
1.运行环境
python3.6
pytorch
pandas
transformers==2.8.0
sklearn
tqdm

2.下载bert的预训练模型
放在当前目录下，修改一下代码中的路径。

3.bert+wmd
模型过程：将问题对分开输入到同一个bert（权重共享）进行编码：  
cls + query + seq 
cls + question + seq 
获取cls输出的当做句子向量，使用句向量进行计算余弦相似度。
bert最后一层的输出每个位置的向量当做句子中每个词的词向量，用来计算wmd。

4.bert_cos.py
这个文件是使用余弦相似度计算问题对的相似度的，不是wmd，可以用来与wmd做对比。

5.数据集
本项目使用的数据集是CCKS 2018 微众银行智能客服问句匹配大赛-数据集  
存放在data目录下

