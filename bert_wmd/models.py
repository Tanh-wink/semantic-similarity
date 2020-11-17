import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F
from transformers import BertPreTrainedModel

class FaqModel(BertPreTrainedModel):
    """
    Model class that combines a pretrained bert model with a linear later
    """

    def __init__(self, conf, pretrained_model_path=None):
        super(FaqModel, self).__init__(conf)
        # Load the pretrained BERT model
        self.bert = transformers.BertModel.from_pretrained(pretrained_model_path, config=conf)
        # Set 10% dropout to be applied to the BERT backbone's output
        self.drop_out = nn.Dropout(0.5)
        # 768 is the dimensionality of bert-base-uncased's hidden representations
        self.classier = nn.Linear(768, 2)
        torch.nn.init.normal_(self.classier.weight, std=0.02)

    def forward(self, input_ids, input_mask, token_type_ids):
        # Return the hidden states from the BERT backbone
        sequence_output, pooled_output, encoder_outputs = self.bert(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids
        )  # bert_layers x bs x SL x (768 * 2)

        # Apply 10% dropout to the last 2 hidden states
        pooled_output = self.drop_out(pooled_output)  # bs x SL x (768 * 2)
        # The "dropped out" hidden vectors are now fed into the linear layer to output two scores
        logits = self.classier(pooled_output)  # bs x SL x 2

        return logits


class SiameseWmdModel(BertPreTrainedModel):
    """
    Model class that combines a pretrained bert model with a linear later
    """

    def __init__(self, conf, pretrained_model_path=None):
        super(SiameseWmdModel, self).__init__(conf)
        # Load the pretrained BERT model
        self.bert = transformers.BertModel.from_pretrained(pretrained_model_path, config=conf)

    def forward_onece(self, input_ids, input_mask, token_type_ids):
        # Return the hidden states from the BERT backbone
        sequence_output, pooled_output, encoder_outputs = self.bert(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids
        )  # bert_layers x bs x SL x (768 * 2)
        return sequence_output, pooled_output, encoder_outputs

    def forward(self, query_ids, query_mask, query_type_ids, question_ids, question_mask, question_type_ids):
        # Return the hidden states from the BERT backbone
        query_sequence, query_pooled, query_encoder = self.forward_onece(
            query_ids,
            input_mask=query_mask,
            token_type_ids=query_type_ids
        )  #
        question_sequence, question_pooled, question_encoder = self.forward_onece(
            question_ids,
            input_mask=question_mask,
            token_type_ids=question_type_ids
        )  #

        return query_sequence, question_sequence, query_pooled, question_pooled