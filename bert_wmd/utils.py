import numpy as np
import torch
import torch.nn.functional as F
import pickle
from pyemd import emd
from numpy import float64
from torch.nn import CrossEntropyLoss

from scipy.optimize import linprog


# def wmdistance(document1, document2):
#     def nbow(document):
#         d = zeros(vocab_len, dtype=double)
#         nbow = dictionary.doc2bow(document)  # Word frequencies.
#         doc_len = len(document)
#         for idx, freq in nbow:
#             d[idx] = freq / float(doc_len)  # Normalized word frequencies.
#         return d
#
#     # Compute nBOW representation of documents.
#     d1 = nbow(document1)
#     d2 = nbow(document2)

def batch_wmdistance(query_vec, question_vec, query, questions):
    """WMD（Word Mover's Distance）
    x.shape=[m,d], y.shape=[n,d]
    """
    q_Q_wmd = []
    batch = len(query)
    for i in range(batch):
        query_len = len(query[i])
        question_len = len(questions[i])
        q_Q_wmd.append(wmdistance(query_vec[i][:query_len], question_vec[i][:question_len]))
    return q_Q_wmd


def wmdistance(sent1, sent2):
    """WMD（Word Mover's Distance）
    x.shape=[m,d], y.shape=[n,d]
    """
    p = np.ones(sent1.shape[0], dtype=np.float64) / sent1.shape[0]
    q = np.ones(sent2.shape[0], dtype=np.float64) / sent2.shape[0]
    D = float64(np.sqrt(np.square(sent1[:, None] - sent2[None, :]).mean(axis=2)))
    return wasserstein_distance(p, q, D)


def wasserstein_distance(p, q, D):
    """通过线性规划求Wasserstein距离
    p.shape=[m], q.shape=[n], D.shape=[m, n]
    p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]
    """
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = D.reshape(-1)
    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    return result.fun


# def wmdistance1(self, document1, document2):
#
#     # If pyemd C extension is available, import it.
#     # If pyemd is attempted to be used, but isn't installed, ImportError will be raised in wmdistance
#
#
#     # Remove out-of-vocabulary words.
#     len_pre_oov1 = len(document1)
#     len_pre_oov2 = len(document2)
#     document1 = [token for token in document1 if token in self]
#     document2 = [token for token in document2 if token in self]
#     diff1 = len_pre_oov1 - len(document1)
#     diff2 = len_pre_oov2 - len(document2)
#     if diff1 > 0 or diff2 > 0:
#         logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)
#
#     if not document1 or not document2:
#         logger.info(
#             "At least one of the documents had no words that were in the vocabulary. "
#             "Aborting (returning inf)."
#         )
#         return float('inf')
#
#     dictionary = Dictionary(documents=[document1, document2])
#     vocab_len = len(dictionary)
#
#     if vocab_len == 1:
#         # Both documents are composed by a single unique token
#         return 0.0
#
#     # Sets for faster look-up.
#     docset1 = set(document1)
#     docset2 = set(document2)
#
#     # Compute distance matrix.
#     distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
#     for i, t1 in dictionary.items():
#         if t1 not in docset1:
#             continue
#
#         for j, t2 in dictionary.items():
#             if t2 not in docset2 or distance_matrix[i, j] != 0.0:
#                 continue
#
#             # Compute Euclidean distance between word vectors.
#             distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(np.sum((self[t1] - self[t2])**2))
#
#     if np.sum(distance_matrix) == 0.0:
#         # `emd` gets stuck if the distance matrix contains only zeros.
#         print('The distance matrix is all zeros. Aborting (returning inf).')
#         return float('inf')
#
#     def nbow(document):
#         d = zeros(vocab_len, dtype=double)
#         nbow = dictionary.doc2bow(document)  # Word frequencies.
#         doc_len = len(document)
#         for idx, freq in nbow:
#             d[idx] = freq / float(doc_len)  # Normalized word frequencies.
#         return d
#
#     # Compute nBOW representation of documents.
#     d1 = nbow(document1)
#     d2 = nbow(document2)
#
#     # Compute WMD.
#     return emd(d1, d2, distance_matrix)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # loss_contrastive = torch.mean(label.float() * torch.pow(distance, 2)) / 2 + \
        #                    (1 - label.float()) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss_contrastive = torch.mean(label.float() * torch.pow(distance, 2) + (1.0 - label.float()) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss_contrastive / 2

def load_pkl_data(filePath):
    print(f'load {filePath}')
    with open(filePath, 'rb') as fp:
        data_pkl = fp.read()
    return pickle.loads(data_pkl)

def save_pkl_data(data, filePath):
    print(f'save {filePath}')
    data_pkl = pickle.dumps(data)
    with open(filePath, 'wb') as fp:
        fp.write(data_pkl)


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(model_path)
        self.val_score = epoch_score


class ModelSaver:
    def __init__(self, mode="max"):
        self.counter = 0
        self.mode = mode
        self.best_score = None
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path, step='', epoch=''):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path, step=step, epoch=epoch)

        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path, step=step, epoch=epoch)

    def save_checkpoint(self, epoch_score, model, model_path, step='', epoch=''):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Epoch:{}, step:{} ,Validation score improved ({} --> {}). Saving model!'.format(epoch, step, self.val_score, epoch_score))
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(model_path)
        self.val_score = epoch_score


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))