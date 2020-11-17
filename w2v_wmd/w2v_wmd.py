import logging
import os
import jieba
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from gensim.similarities import WmdSimilarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from trainW2V import get_data, preprocess


logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

w2v_config = {
    "embeddingSize" : 300,
    "trainIter": 1000,
    "min_count": 1
}

data_dir = "../data/"
corpus_path = "corpus_forW2V.txt"
model_path = data_dir + 'word2vec_{}dim_{}iters.w2v.model'.format(w2v_config["embeddingSize"], w2v_config["trainIter"])


def load_w2v(model_path):
    logger.info("********* loading pretrained word2vec model ***********")
    model = Word2Vec.load(model_path)
    logger.info("loaded pretrained word2vec model successfully")
    return model


def evaluate(true_labels, pred_labels):
    acc = accuracy_score(true_labels, pred_labels)
    P = precision_score(true_labels, pred_labels)
    R = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    return acc, P, R, f1


def query(q, instance):
    sims = instance[q]  # np.array

    return sims

if __name__ == '__main__':

    train_all_file = "atec_nlp_sim_train_all.csv"
    train_all = get_data(data_dir + train_all_file)
    if not os.path.exists(data_dir+"train_all_preprocessed.csv"):
        train_all = preprocess(train_all)
        train_all.to_csv(data_dir + "train_all_preprocessed.csv", sep='\t', header=None, index=None, encoding='utf-8')
    else:
        train_all = pd.read_csv(data_dir+"train_all_preprocessed.csv",
                                sep="\t", header=None, names=["id", "q1", "q2", "label", "q1_seg", "q2_seg"])

    q1 = train_all["q1"].tolist()
    q2 = train_all["q2"].tolist()
    q1_segs = [line.strip().split(" ") for line in train_all["q1_seg"].tolist()]
    q2_segs = [line.strip().split(" ") for line in train_all["q2_seg"].tolist()]
    labels = [line for line in train_all["label"].tolist()]
    #
    thresholds = [0.001, 0.002, 0.005, 0.01]
    best_threshold = 0
    best_f1 = 0

    n_best = 10
    w2v_model = load_w2v(model_path)
    instance = WmdSimilarity(q2_segs, w2v_model, num_best=n_best)

    logger.info("******** evaluate ********")


    # # 在相似性类中的“查找”query


    topk_sims = query(q1_segs[2], instance)
    threshold = 0.5
    logger.info(f"query:{q1[2]}")

    i = 1
    for index, sim in topk_sims[:10]:
        if sim >= threshold:
            logger.info(f"{i}: similar question:{q2[index]}, sim:{sim}")
            i += 1

    # if topk_sims[0][1] < threshold:
    #     logger.info("This query has 0 similar question in the Q&A set")
    # else:
    #     i = 1
    #     for index, sim in topk_sims:
    #         if sim >= threshold:
    #             logger.info(f"{i}: similar question:{q2[index]}, sim:{sim}")
    #             i += 1



    # 使用搜狗金融word2vec词向量
    # word_vectors_path = data_dir + "word_embeddings/sgns.sogou.bigram"
    # word_vectors = KeyedVectors.load_word2vec_format(word_vectors_path)
    #

    # for threshold in thresholds:
    #     pred_labels = []
    #     for q1_seg, q2_seg in zip(q1_segs, q2_segs):
    #         q1_q2_sim = word_vectors.wmdistance(q1_seg, q2_seg)
    #         if q1_q2_sim > threshold:
    #             pred_label = 1
    #         else:
    #             pred_label = 0
    #         pred_labels.append(pred_label)
    #     acc, P, R, f1 = evaluate(labels, pred_labels)
    #     if f1 > best_f1:
    #         best_f1 = f1
    #         best_threshold = threshold
    #     logger.info("threshold:{}, acc:{:.4f}, P:{:.4f}, R:{:.4f}, f1:{:.4f}".format(threshold, acc, P, R, f1))
    #
    # logger.info("best threshold is : {}, with best f1 : {}".format(best_threshold, best_f1))

    # m = 10
    # sims = get_similarities(q1_seg[50:60], q2_seg, word_vectors, n_best)
    # # # 返回相似结果
    # for i in range(m):
    #     logger.info(i+1)
    #     logger.info('q1:{}   q2:{}    label:{}'.format(train_all["q1"][i], train_all["q2"][i], train_all["label"][i]))
    #     logger.info("best_sim_q2:{}    wmd:{}".format(train_all["q2"][sims[i][0][0]], sims[i][0][1]))
    # for i in range(n_best):
    #     print('sim = {}' .format(sims[i]))
        # print(q2[sims[i][0]])
