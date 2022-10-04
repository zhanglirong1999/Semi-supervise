# Importing stock libraries
from cProfile import label
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import sys
sys.path.append(os.getcwd())
import logging

logger = logging.getLogger()

logger.setLevel(logging.INFO)


class TupleDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length, sep_token=" ", model="roberta", is_eval=False):
        # infer file dataset given a certain relation
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length
        
        self.anchor = self.data["anchor"]
        self.positive = self.data["positive"]
        self.negative = self.data["negative"]

        self.anchor_label = self.data["anchor_label"]
        self.positive_label = self.data["positive_label"]
        self.negative_label = self.data["negative_label"]           

        self.sep_token = sep_token
        self.model = model
        self.is_eval = is_eval

        # for evaluation set only
        self.clss = pd.Series(["" for i in range(len(self.data))]) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor = str(self.anchor[index])
        positive = str(self.positive[index])
        negative = str(self.negative[index])

        anchor_label = float(self.anchor_label[index])
        positive_label = float(self.positive_label[index])
        negative_label = float(self.negative_label[index])  

    
        # print(text)
        source_anchor = self.tokenizer(anchor, padding='max_length', max_length=self.max_length, return_tensors='pt', truncation=True)
        source_positive = self.tokenizer(positive, padding='max_length', max_length=self.max_length, return_tensors='pt', truncation=True)
        source_negative = self.tokenizer(negative, padding='max_length', max_length=self.max_length, return_tensors='pt', truncation=True)

        source_ids_anchor = source_anchor['input_ids'].squeeze()
        source_mask_anchor = source_anchor['attention_mask'].squeeze()

        source_ids_positive = source_positive['input_ids'].squeeze()
        source_mask_positive = source_positive['attention_mask'].squeeze()

        source_ids_negative = source_negative['input_ids'].squeeze()
        source_mask_negative = source_negative['attention_mask'].squeeze()

        return {
            'ids_anchor': source_ids_anchor.to(dtype=torch.long),
            'mask_anchor': source_mask_anchor.to(dtype=torch.long), 
            'label_anchor': torch.tensor(anchor_label).to(dtype=torch.float),
            'ids_positive': source_ids_positive.to(dtype=torch.long),
            'mask_positive': source_mask_positive.to(dtype=torch.long), 
            'label_positive': torch.tensor(positive_label).to(dtype=torch.float),
            'ids_negative': source_ids_negative.to(dtype=torch.long),
            'mask_negative': source_mask_negative.to(dtype=torch.long), 
            'label_negative': torch.tensor(negative_label).to(dtype=torch.float),
        }
