import os
import re
import jieba
import logging
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import json
import pickle
from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# file path
data_dir = "../data/"
train_file = "atec_nlp_sim_train_0.6.csv"
train_all_file = "atec_nlp_sim_train_all.csv"
test_file = "atec_nlp_sim_test_0.4.csv"
stop_words_file = 'stop_words.txt'
dict_file = 'dict_all.txt'
spelling_corrections_file = 'spelling_corrections.json'

jieba.load_userdict(data_dir+"dict_all.txt")

def get_data(path):
    return pd.read_csv(path, sep="\t", header=None, names=["id", "q1", "q2", "label"])

def load_stopwords(path):
    stopwords = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin.readlines():
            word = line.strip()
            stopwords.append(word)
        return stopwords


def load_spelling_corrections(path):
    with open(path, "r", encoding="utf-8") as fin:
        spelling_corrections = json.load(fin)
        return spelling_corrections


def transform_other_word(sence, corr_dict):
    for token_str, replac_str in corr_dict.items():
        sence = sence.replace(token_str, replac_str)
    return sence

def seg_sentence(sentence,stop_words):
    """
    对句子进行分词
    :param sentence:句子，String
    """
    mark = ["。", "？"]
    sentence_seged = jieba.cut(sentence.strip())
    out_str = ""
    for word in sentence_seged:
        if word not in stop_words:
            if word != " ":
                out_str += word
                out_str += " "
    out_str = out_str[:-1]
    if out_str[-1] not in mark:
        out_str += mark[1]
    return out_str


def save(data):
    pass


def preprocess(train_all):
    # 加载停用词
    stopwords = load_stopwords(data_dir + stop_words_file)
    # 加载拼写错误替换词
    spelling_corrections = load_spelling_corrections(data_dir + spelling_corrections_file)

    re_object = re.compile(r'\*+')  # 去除句子中的脱敏数字***，替换成一
    word_frequence = defaultdict(int)  # 记录词汇表词频
    train_all.fillna("null")
    train_all.dropna(axis=0, how="any", subset=["q1", "q2"])
    with tqdm(total=len(train_all)) as pbar:
        pbar.set_description("preprocessing train all:")
        for index, row in train_all.iterrows():

            # 分别遍历每行的两个句子，并进行分词处理
            for col_name in ["q1", "q2"]:
                # 替换掉脱敏的数字
                re_str = re_object.subn(u"十一", row[col_name])
                # 纠正一些词
                spell_corr_str = transform_other_word(re_str[0], spelling_corrections)
                # 分词
                seg_str = seg_sentence(spell_corr_str, stopwords)
                # 计算词频
                for word in seg_str.split(" "):
                    word_frequence[word] = word_frequence[word] + 1

                train_all.at[index, col_name+"_seg"] = seg_str
            pbar.update(1)

    with open(data_dir + "word_frequence.pkl", "wb") as fout:
        pickle.dump(word_frequence, fout)
    logger.info("There are  {} words in vocabs".format(len(word_frequence)))
    logger.info("There are  {} examples in train_all".format(train_all.shape[0]))
    word_frequence_df = pd.DataFrame({"word_frequent":list(word_frequence.values())}, index=list(word_frequence.keys()))
    logger.info(word_frequence_df.describe(percentiles=[0.5,0.8,0.9,0.99,0.999]))
    return train_all


def buildCorpusForW2V(data, saveFilePath):
    corpus = []
    q1 = [line.strip().split(" ") for line in data["q1_seg"].tolist()]
    q2 = [line.strip().split(" ") for line in data["q2_seg"].tolist()]
    for sent in q1:
        corpus.extend(sent)

    for sent in q2:
        corpus.extend(sent)

    corpusWordNum = len(set(corpus))
    logger.info('该语料一共有 {} 个词，总词量：{}'.format(corpusWordNum, len(corpus)))
    with open(saveFilePath, 'w', encoding='utf-8') as fp:
        for item in corpus:
            fp.write('{} '.format(item))
    logger.info('save {} done'.format(os.path.abspath(saveFilePath)))

def trainW2V(path, embeddingSize=150, trainIter=5, min_count=5):
    Word2VecParam = {
        'sg': 1,
        'size': embeddingSize,
        'window': 5,
        'min_count': min_count,
        'iter': trainIter,  # 迭代次数
        # 'negative': 3,
        # 'sample': 0.001,
        'hs': 1,
        'workers': 4,
    }
    basePath = os.path.split(path)[0]

    sentences = word2vec.Text8Corpus('{}'.format(path))
    model = word2vec.Word2Vec(sentences, **Word2VecParam)
    print('word2vec model training done')
    print('word2vec model：{}'.format(model))
    savePath = '{}/word2vec_{}dim_{}iters.w2v.model'.format(basePath, embeddingSize, trainIter)
    model.save(savePath)
    print('{} save done'.format(os.path.abspath(savePath)))
    vocabulary = {}
    for vocab in model.wv.vocab:
        vector = model[vocab]
        vocabulary[vocab] = vector.tolist()
    logger.info("vocab size:{}".format(len(vocabulary)))
    savePath2 = '{}/word2vec_{}dim_{}iters.w2v.vocab.json'.format(basePath, embeddingSize, trainIter)
    with open('{}'.format(savePath2), 'w',
              encoding='utf-8') as fp:
        vocabularyJson = json.dumps(vocabulary)
        fp.write(vocabularyJson)
    print('{} save done'.format(savePath2))
    return model

if __name__ == '__main__':

    train_all = get_data(data_dir + train_all_file)
    train_all = preprocess(train_all)
    train_all.to_csv(data_dir + "train_all_preprocessed.csv", sep='\t', header=None, index=None, encoding='utf-8')
    buildCorpusForW2V(train_all, data_dir + 'corpus_forW2V.txt')
    trainW2V(
        data_dir + 'corpus_forW2V.txt',
        embeddingSize=300,
        trainIter=10**3,
        min_count=3
    )




