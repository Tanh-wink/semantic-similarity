import os
import logging
import torch
import pandas as pd
import warnings

import tokenizers
import transformers
from tqdm.autonotebook import tqdm

from utils import load_pkl_data, save_pkl_data
from utils import ContrastiveLoss, batch_wmdistance

from models import FaqModel, SiameseWmdModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class config:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    # EPOCHS = 5
    # NUM_LABEL = 2
    THRESHOLDE = 0.5

    BERT_PATH = "./chinese_wwm_pytorch" #"./chinese_wwm_pytorch"
    TRAIN_FILE = "../data/dataForBert/train.csv"
    DEV_FILE = "../data/dataForBert/dev.csv"
    TEST_FILE = "../data/dataForBert/test.csv"
    TOKENIZER = tokenizers.BertWordPieceTokenizer(
        f"{BERT_PATH}/vocab.txt",
        lowercase=True
    )
    train_features = '../data/dataForBert/train_features_wmd.pkl'
    valid_features = '../data/dataForBert/valid_features_wmd.pkl'
    MODEL_SAVE_PATH = f"../output/trained_wmd_{BERT_PATH.split('/')[-1]}"
    PREDICT_FILE_SAVE_PATH = f"../output/"


if not os.path.exists(config.MODEL_SAVE_PATH):
    os.makedirs(config.MODEL_SAVE_PATH)


class SiameseDataset:
    """
    Dataset which stores the tweets and returns them as processed features
    """

    def __init__(self, query, question, label=None):
        self.query = query
        self.question = question
        self.label = label
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.query)

    def __getitem__(self, item):
        query = self.query[item]
        question = self.question[item]
        query_ids, query_mask, query_type_ids = self.process_data(query)
        question_ids, question_mask, question_type_ids = self.process_data(question)

        if self.label is not None:
            label = self.label[item]
            # Return the processed data where the lists are converted to `torch.tensor`s
            return {
                "query": query,
                "question": question,
                'query_ids': torch.tensor(query_ids, dtype=torch.long),
                'query_mask': torch.tensor(query_mask, dtype=torch.long),
                'query_type_ids': torch.tensor(query_type_ids, dtype=torch.long),
                'question_ids': torch.tensor(question_ids, dtype=torch.long),
                'question_mask': torch.tensor(question_mask, dtype=torch.long),
                'question_type_ids': torch.tensor(question_type_ids, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'query_ids': torch.tensor(query_ids, dtype=torch.long),
                'query_mask': torch.tensor(query_mask, dtype=torch.long),
                'token_type_ids': torch.tensor(query_type_ids, dtype=torch.long),
                'question_ids': torch.tensor(question_ids, dtype=torch.long),
                'question_mask': torch.tensor(question_mask, dtype=torch.long),
                'question_type_ids': torch.tensor(question_type_ids, dtype=torch.long),
            }

    def process_data(self, q):
        input_ids = self.tokenizer.encode(q).ids

        if len(input_ids) > self.max_len:
            input_ids = input_ids[:-1][:self.max_len - 1] + [102]

        token_type_ids = [0] * len(input_ids)
        input_masks = [1] * len(token_type_ids)

        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            input_masks = input_masks + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == self.max_len
        assert len(input_masks) == self.max_len
        assert len(token_type_ids) == self.max_len

        return input_ids, input_masks, token_type_ids

def run_one_step(batch, model, device):
    query_ids = batch["query_ids"]
    query_type_ids = batch["query_type_ids"]
    query_mask = batch["query_mask"]
    question_ids = batch["question_ids"]
    question_type_ids = batch["question_type_ids"]
    question_mask = batch["question_mask"]
    # Move ids, masks, and targets to gpu while setting as torch.long
    query_ids = query_ids.to(device, dtype=torch.long)
    query_type_ids = query_type_ids.to(device, dtype=torch.long)
    query_mask = query_mask.to(device, dtype=torch.long)
    question_ids = question_ids.to(device, dtype=torch.long)
    question_type_ids = question_type_ids.to(device, dtype=torch.long)
    question_mask = question_mask.to(device, dtype=torch.long)

    # Reset gradients
    model.zero_grad()
    # Use ids, masks, and token types as input to the model
    # Predict logits for each of the input tokens for each batch
    query_sequence, question_sequence, query_pooled, question_pooled = model(query_ids=query_ids, query_mask=query_mask, query_type_ids=query_type_ids, question_ids=question_ids, question_mask=question_mask, question_type_ids=question_type_ids)  #

    return query_sequence, question_sequence, query_pooled, question_pooled


def predict(data_loader, model, device, threshold):
    """
    Trains the bert model on the twitter data
    """
    # Set model to training mode (dropout + sampled batch norm is activated)
    pred_labels = []
    labels = []
    wmd = []
    model.eval()

    # Set tqdm to add loading screen and set the length
    tk0 = tqdm(data_loader, total=len(data_loader))
    # Train the model on each batch
    for bi, batch in enumerate(tk0):
        querys = batch["query"]
        questions = batch["question"]

        query_sequence, question_sequence, query_pooled, question_pooled = run_one_step(batch, model, device)
        label = batch["label"].numpy().tolist()
        wmd_distance = batch_wmdistance(query_sequence.detach().cpu().numpy(), question_sequence.detach().cpu().numpy(),
                                        querys, questions)

        pred_label = [1 if wmd > threshold else 0 for wmd in wmd_distance]
        # print(f"label:{label}")
        # print(f"pred_label:{pred_label}")
        pred_labels.extend(pred_label)
        labels.extend(label)
        wmd.extend(wmd_distance)
        # Calculate the jaccard score based on the predictions for this batch
    acc, f1, auc = calculate_metrics_score(
        label=labels,
        pred_label=pred_labels,
        cal_auc=True
    )
    return pred_labels, wmd, acc, f1, auc


def calculate_metrics_score(label, pred_label, cal_auc=False):
    """
    Calculate the jaccard score from the predicted span and the actual span for a batch of tweets
    """
    acc = accuracy_score(label, pred_label)
    f1 = f1_score(label, pred_label)
    if cal_auc:
        auc_score = roc_auc_score(label, pred_label)
        return acc, f1, auc_score
    else:

        return acc, f1


def run():
    """
    Train model for a speciied fold
    """
    # Read train csv and dev csv
    df_train = pd.read_csv(config.TRAIN_FILE)
    df_valid = pd.read_csv(config.DEV_FILE)

    # Instantiate TweetDataset with training data
    train_dataset = SiameseDataset(
        query=df_train.sentence1.values,
        question=df_train.sentence2.values,
        label=df_train.label.values
    )

    if os.path.exists(config.train_features):
        train_dataset = load_pkl_data(config.train_features)
    else:
        train_dataset = [item for item in train_dataset]
        save_pkl_data(train_dataset, config.train_features)

    # Instantiate DataLoader with `train_dataset`
    # This is a generator that yields the dataset in batches
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=config.TRAIN_BATCH_SIZE
    )

    # Instantiate TweetDataset with validation data
    valid_dataset = SiameseDataset(
        query=df_valid.sentence1.values,
        question=df_valid.sentence2.values,
        label=df_valid.label.values,

    )

    if os.path.exists(config.valid_features):
        valid_dataset = load_pkl_data(config.valid_features)
    else:
        valid_dataset = [item for item in valid_dataset]
        save_pkl_data(valid_dataset, config.valid_features)

    # Instantiate DataLoader with `valid_dataset`
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False
    )

    # Set device as `cuda` (GPU)
    device = torch.device("cuda")
    # Load pretrained BERT (bert-base-uncased)
    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
    # Output hidden states
    # This is important to set since we want to concatenate the hidden states from the last 2 BERT layers
    model_config.output_hidden_states = True
    # Instantiate our model with `model_config`
    model = SiameseWmdModel(conf=model_config, pretrained_model_path=config.BERT_PATH)
    # Move the model to the GPU
    model.to(device)

    # I'm training only for 3 epochs even though I specified 5!!!
    pred_labels, wmd, acc, f1, auc = predict(train_data_loader, model, device)
    logger.info(f"train set : acc = {acc}, f1 score = {f1}, auc = {auc}" )
    df_train["pred_label"] = pred_labels
    df_train["wmd"] = wmd
    df_train.to_csv("../output/train_predict.csv")

    thresholds = [0.25, 0.23]
    best_f1 = 0
    best_th = 0
    for threshold in thresholds:
        pred_labels, wmd, acc, f1, auc = predict(valid_data_loader, model, device, threshold)
        logger.info(f"dev set :threshold={threshold}  acc = {acc}, f1 score = {f1}, auc = {auc}")

        if f1 > best_f1:
            best_f1 = f1
            best_th = threshold
    print(f"best threshold: {best_th} with best f1 {best_f1}")

    df_valid["pred_label"] = pred_labels
    df_valid["wmd"] = wmd
    df_valid.to_csv("../output/dev_predict.csv")

if __name__ == '__main__':
    run()


