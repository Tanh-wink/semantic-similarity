# word2vec+wmd
1. 运行环境  
python3.6  
gensim  
jieba  
pandas  
numpy  
sklearn  

2.data目录下是存放数据
本项目是使用2018年atec蚂蚁金服NLP智能客服比赛的数据进行实验。  
https://dc.cloud.alipay.com/index#/topic/intro?id=3  
如果要去停用词， 还需要自行去下载停用词表放在该目录下。

3.文件说明  
trainW2V.py：该文件是data里的所有数据使用gensim库中的word2vec模型进行训练词向量的代码
w2v_wmd.py：使用已经训练好的word2vec词向量进行计算每一问题对的wmd相似度。

若需要使用别人已经训练好的词向量可以到https://github.com/Embedding/Chinese-Word-Vectors下载，有非常多种的中文词向量可供下载。  
经过实验对比，外部词向量虽然使用训练的语料库非常丰富，以及词向量语义更好，但是与我们的实验数据不是同一个领域的，效果不及我们自己使用实验数据训练出来的词向量。

4. 进行搬运代码时需要注意修改一下路径问题。
