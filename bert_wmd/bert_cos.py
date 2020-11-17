import os
import logging
import torch
import pandas as pd
import numpy as np

from torch.optim import lr_scheduler
import torch.nn.functional as F
import tokenizers
import transformers
from transformers import AdamW,BartForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm

import utils
from utils import load_pkl_data, save_pkl_data
from utils import ContrastiveLoss, batch_wmdistance

from models import FaqModel, SiameseWmdModel
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class config:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    EPOCHS = 5
    NUM_LABEL = 2
    THRESHOLDE = 0.1

    SEED = 41

    DO_TRAIN = True
    DO_TEST = True

    BERT_PATH = "./chinese_wwm_pytorch"
    TRAIN_FILE = "../data/dataForBert/train.csv"
    DEV_FILE = "../data/dataForBert/dev.csv"
    TEST_FILE = "../data/dataForBert/test.csv"
    TOKENIZER = tokenizers.BertWordPieceTokenizer(
        f"{BERT_PATH}/vocab.txt",
        lowercase=True
    )
    train_features = '../data/dataForBert/train_features_wmd.pkl'
    valid_features = '../data/dataForBert/valid_features_wmd.pkl'
    MODEL_SAVE_PATH = f"../output/trained_cos_{BERT_PATH.split('/')[-1]}"
    PREDICT_FILE_SAVE_PATH = f"../output/"


if not os.path.exists(config.MODEL_SAVE_PATH):
    os.makedirs(config.MODEL_SAVE_PATH)


def seed_set(seed):
    '''
    set random seed of cpu and gpu
    :param seed:
    :return:
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cal_distance(sent1, sent2, cos=False):
    if cos:
        cosine_similarity = F.cosine_similarity(sent1, sent2)
        return cosine_similarity
    else:
        euclidean_distance = F.pairwise_distance(sent1, sent2, keepdim=True)
        return euclidean_distance


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
    query_sequence, question_sequence, query_pooled, question_pooled = model(query_ids=query_ids, query_mask=query_mask, query_type_ids=query_type_ids,
                   question_ids=question_ids, question_mask=question_mask, question_type_ids=question_type_ids)  #

    return query_sequence, question_sequence, query_pooled, question_pooled


def train_fn(data_loader, model, optimizer, device, scheduler=None, threshold=None):
    """
    Trains the bert model on the twitter data
    """
    # Set model to training mode (dropout + sampled batch norm is activated)
    model.train()

    # Set tqdm to add loading screen and set the length
    tk0 = tqdm(data_loader, total=len(data_loader))
    # Train the model on each batch
    for bi, batch in enumerate(tk0):

        query_sequence, question_sequence, query_pooled, question_pooled = run_one_step(batch, model, device)
        labels = batch["label"].to(device)
        distance = cal_distance(query_pooled, question_pooled, cos=False)
        # Calculate batch loss based on CrossEntropy
        loss_fn = ContrastiveLoss(margin=1)
        loss = loss_fn(distance, labels)
        # Calculate gradients based on loss
        loss.backward()
        # Adjust weights based on calculated gradients
        optimizer.step()
        # Update scheduler
        scheduler.step()

        pred_labels = [1 if d > threshold else 0 for d in distance]
        # Calculate the jaccard score based on the predictions for this batch
        acc, f1 = calculate_metrics_score(
            label=labels.cpu().numpy(),
            pred_label=np.array(pred_labels),
        )
        # Print the average loss and jaccard score at the end of each batch
        tk0.set_postfix(loss=loss.item(), acc=acc, f1=f1)


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


def eval_fn(data_loader, model, device, test=False):
    """
    Evaluation function to predict on the test set
    """
    # Set model to evaluation mode
    # I.e., turn off dropout and set batchnorm to use overall mean and variance (from training), rather than batch level mean and variance
    # Reference: https://github.com/pytorch/pytorch/issues/5406
    model.eval()
    true_labels = []
    pred_labels = []
    # Turns off gradient calculations (https://datascience.stackexchange.com/questions/32651/what-is-the-use-of-torch-no-grad-in-pytorch)
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        # Make predictions and calculate loss / acc, f1 score for each batch
        for bi, batch in enumerate(tk0):

            query_sequence, question_sequence, query_pooled, question_pooled = run_one_step(batch, model, device)

            labels = batch["label"].to(device)

            # Calculate loss for the batch
            distance = cal_distance(query_pooled, question_pooled, cos=False)
            # Calculate batch loss based on CrossEntropy
            loss_fn = ContrastiveLoss(margin=1)
            loss = loss_fn(distance, labels)

            # Apply softmax to the predicted logits
            # This converts the "logits" to "probability-like" scores
            pred_label = [1 if d > config.THRESHOLDE else 0 for d in distance]
            labels = labels.cpu().numpy().tolist()
            pred_labels.extend(pred_label)
            true_labels.extend(labels)
            acc, f1 = calculate_metrics_score(
                label=labels,
                pred_label=pred_label,
            )
            # Print the running average loss and acc and f1 score
            tk0.set_postfix(loss=loss.item(), acc=acc, f1=f1)

    acc, f1, auc = calculate_metrics_score(
        label=true_labels,
        pred_label=pred_labels,
        cal_auc=True
    )
    logger.info(f"acc = {acc}, f1 = {f1}, auc={auc}")
    return acc, f1, auc


def train():
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
        shuffle=True,
        batch_size=config.TRAIN_BATCH_SIZE
    )

    # Instantiate TweetDataset with validation data
    valid_dataset = SiameseDataset(
        query=df_valid.sentence1.values,
        question=df_valid.sentence2.values,
        label=df_valid.label.values
    )

    if os.path.exists(config.valid_features):
        valid_dataset = load_pkl_data(config.valid_features)
    else:
        valid_dataset = [item for item in valid_dataset]
        save_pkl_data(valid_dataset, config.valid_features)

    # Instantiate DataLoader with `valid_dataset`
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE
    )

    # Set device as `cuda` (GPU)
    device = torch.device("cuda:2")
    # Load pretrained BERT (bert-base-uncased)
    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
    # Output hidden states
    # This is important to set since we want to concatenate the hidden states from the last 2 BERT layers
    model_config.output_hidden_states = True
    # Instantiate our model with `model_config`
    model = SiameseWmdModel(conf=model_config, pretrained_model_path=config.BERT_PATH)
    # Move the model to the GPU
    model.to(device)

    # Calculate the number of training steps
    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    # Get the list of named parameters
    param_optimizer = list(model.named_parameters())
    # Specify parameters where weight decay shouldn't be applied
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # Define two sets of parameters: those with weight decay, and those without
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    # Instantiate AdamW optimizer with our two sets of parameters, and a learning rate of 3e-5
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    # Create a scheduler to set the learning rate at each training step
    # "Create a schedule with a learning rate that decreases linearly after linearly increasing during a warmup period." (https://pytorch.org/docs/stable/optim.html)
    # Since num_warmup_steps = 0, the learning rate starts at 3e-5, and then linearly decreases at each training step
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # Apply early stopping with patience of 2
    # This means to stop training new epochs when 2 rounds have passed without any improvement
    es = utils.EarlyStopping(patience=2, mode="max")

    thresholds = [0.1, 0.15, 0.20]
    best_f1 = 0
    best_th = 0
    for threshold in thresholds:

        # I'm training only for 3 epochs even though I specified 5!!!
        for epoch in range(config.EPOCHS):
            train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler, threshold=threshold)
            acc, f1, auc = eval_fn(valid_data_loader, model, device)

            # logger.info(f"acc = {acc}, f1 score = {f1}")
            es(f1, model, model_path=config.MODEL_SAVE_PATH)
            if es.early_stop:
                if f1 > best_f1:
                    best_f1 = f1
                    best_th = threshold
                print("Early stopping ********")
                break
    logger.info(f"best threshold:{best_th}, best f1 :{best_f1}")




def predict(predict_file):
    df_test = pd.read_csv(config.TEST_FILE)
    # df_test.loc[:, "selected_text"] = df_test.text.values
    device = torch.device("cuda:2")
    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
    model_config.output_hidden_states = True
    # Load each of the five trained models and move to GPU
    model = FaqModel(conf=model_config, pretrained_model_path=config.BERT_PATH)
    model.to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH + "/pytorch_model.bin"))
    model.eval()

    # Instantiate TweetDataset with the test data
    test_dataset = SiameseDataset(
        query=df_test.sentence1.values,
        question=df_test.sentence2.values
    )

    # Instantiate DataLoader with `test_dataset`
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )
    pred_labels = []
    # Turn of gradient calculations
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        # Predict the span containing the sentiment for each batch
        for bi, batch in enumerate(tk0):
            # Predict logits
            query_sequence, question_sequence, query_pooled, question_pooled = run_one_step(batch, model, device)
            distance = cal_distance(query_pooled, question_pooled, cos=False)
            pred_label = [1 if d > config.THRESHOLDE else 0 for d in distance]
            pred_labels.extend(pred_label)

    df_test["pred_label"] = pred_labels
    df_test.to_csv(predict_file)

    if "label" in df_test.columns.tolist():
        true_labels = df_test["label"]
        acc, f1 = calculate_metrics_score(
            label=true_labels,
            pred_label=pred_labels,
            cal_auc=True
        )
        logger.info(f"Test Set : acc = {acc}, f1 = {f1}")


def run():
    seed_set(config.SEED)
    if config.DO_TRAIN:
        train()
    if config.DO_TEST:
        predict_file = config.PREDICT_FILE_SAVE_PATH + "predict.csv"
        predict(predict_file)


if __name__ == '__main__':
    run()


