import random
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoTokenizer, AutoModel, RobertaConfig, RobertaModel)

class RoBERTAClassifier(nn.Module):
    def __init__(self, model_name, dropout=0):
        super().__init__()
        self.config = RobertaConfig.from_pretrained(model_name)
        
        # self.model = AutoModel.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.model_type = self.model.config.model_type
        
        self.emb_size = self.model.config.hidden_size # bart

        self.sigmoid = nn.Sigmoid()

        self.linear = nn.Linear(self.emb_size, 2)

        self.dropout = nn.Dropout(p=dropout)

    def get_lm_embedding(self, tokens):
        """
            Input_ids: tensor (num_node, max_length)

            output:
                tensor: (num_node, emb_size)
        """
        outputs = self.model(tokens['input_ids'],
                             attention_mask=tokens['attention_mask'])

        if self.model_type == "bart":
            # embedding of [EOS] in the decoder
            eos_mask = tokens['input_ids'].eq(self.model.config.eos_token_id)

            if torch.any(eos_mask.sum(1) > 1):
                raise ValueError("All examples must have only one <eos> tokens.")
            sentence_representation = outputs[0][eos_mask, :].view(outputs[0].size(0), -1, outputs[0].size(-1))[
                :, -1, :
            ]
        else:
            # embedding of the [CLS] tokens
            sentence_representation = outputs[0][:, 0, :]
        
        return sentence_representation

    def forward(self, tokens):
        """
            tokens: 
        """
        
        embs = self.get_lm_embedding(tokens) # (batch_size, emb_size)
        # embs = embs.view(-1,self.emb_size)
        # x = self.sigmoid(self.linear1(embs)) # (batch_size, 2)
        # x = self.sigmoid(self.linear2(x))
        # x = self.sigmoid(self.linear3(x))
        # x = self.linear4(x)
        # print(x.shape)
        # return x
        # return self.dropout(logits)
        logits = self.linear(embs) # (batch_size, 2)
        # print(logits)
        # logits = torch.softmax(logits, dim=1)         
        # print(logits)
        return self.dropout(logits)