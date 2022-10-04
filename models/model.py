import random
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from transformers import (RobertaTokenizer, AutoModel, RobertaConfig, RobertaModel)
from transformers import RobertaForSequenceClassification
from pytorch_metrics import BinaryAccuracy
from torch.nn import functional as F, CrossEntropyLoss
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score,recall_score ,precision_score

class RoBERTAClassifier(nn.Module):
    def __init__(self, model_name, dropout=0, device='cuda'):
        super().__init__()
        self.config = RobertaConfig.from_pretrained(model_name,
                    num_labels=2
                    )

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification(self.config)
        self.model_type = self.model.config.model_type
        
        self.emb_size = self.model.config.hidden_size # bart

        self.loss_fn = CrossEntropyLoss()
        self.metric = BinaryAccuracy(device).to(device)


        # self.config = RobertaConfig.from_pretrained(model_name)
        
        # # self.model = AutoModel.from_pretrained(model_name)
        # self.model = RobertaModel.from_pretrained(model_name)
        # self.model_type = self.model.config.model_type
        
        # self.emb_size = self.model.config.hidden_size # bart

        self.sigmoid = nn.Sigmoid()

        self.hidden = nn.Linear(self.emb_size, 128)

        self.linear = nn.Linear(128, 2)

        # self.dropout = nn.Dropout(p=dropout)

    def get_lm_embedding(self, tokens):
        """
            Input_ids: tensor (num_node, max_length)

            output:
                tensor: (num_node, emb_size)
        """
        outputs = self.model(tokens['input_ids'],
                             attention_mask=tokens['attention_mask'])
        # print('kkkkkkkkk')
        # print(outputs)
        if self.model_type == "bart":
            # embedding of [EOS] in the decoder
            eos_mask = tokens['input_ids'].eq(self.tokenizer.config.eos_token_id)

            if torch.any(eos_mask.sum(1) > 1):
                raise ValueError("All examples must have only one <eos> tokens.")
            sentence_representation = outputs[0][eos_mask, :].view(outputs[0].size(0), -1, outputs[0].size(-1))[
                :, -1, :
            ]
        else:
            # embedding of the [CLS] tokens
            sentence_representation = outputs[0][:, 0, :]
        # print(sentence_representation)
        return sentence_representation

    def forward(self, tokens):
        """
            tokens: input_ids: 
        """
        y_hat = self.model(tokens['input_ids'],
                             attention_mask=tokens['attention_mask'])
        # print(y_hat)
        return y_hat[0]      
        # embs = self.get_lm_embedding(tokens) # (batch_size, emb_size)
        # print(embs)
        # x = F.relu(self.hidden(embs))

        # logits = self.linear(x) # (batch_size, 2)
        # logits = self.model(tokens['input_ids'],
                            #  attention_mask=tokens['attention_mask']) # (batch_size, 2)
        # print(logists)
        # return self.sigmoid(logits[0])
        return self.sigmoid(logits)

    def compute_metric(self, y_hat, y):
            """
                calculate accuracy
            Args:
                y_hat: model output, y_hat
                y: ground truth label, y
            """
            # print(y_hat)
            # predict_scores = F.softmax(y_hat, dim=1)
            # predict_labels = torch.argmax(predict_scores, dim=-1)
            predict_scores = self.sigmoid(y_hat)
            # print(predict_scores)
            predict_labels = torch.argmax(predict_scores, dim=-1)
            # print('l')
            # print(predict_labels)
            # print(y)
            acc_score = accuracy_score(y_true=y.cpu(), y_pred=predict_labels.cpu())
            # pre =  precision_score(y_true=predict_labels.cpu(), y_pred=y.cpu())
            # recall = recall_score(y_true=predict_labels.cpu(), y_pred=y.cpu())
            f1 = f1_score(y_true=y.cpu(), y_pred=predict_labels.cpu())
            return {"accuracy": acc_score, "f1": f1}
