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


class CKBPDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length, sep_token=" ", model="roberta", is_eval=False):
        # infer file dataset given a certain relation
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length
        if 'quality' in self.data.columns:
            self.quality = self.data['quality']
        else:
            self.quality = self.data['typicality_score']
        self.item_a = self.data["item_a_name"]
        self.item_b = self.data["item_b_name"]
        self.assertion = self.data["assertion"]            

        self.sep_token = sep_token
        self.model = model
        self.is_eval = is_eval

        # for evaluation set only
        self.clss = pd.Series(["" for i in range(len(self.data))]) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item_a = str(self.item_a[index])
        item_b = str(self.item_b[index])
        assertion = str(self.assertion[index])
        text = assertion
        text = text.replace('Item A',item_a)
        text = text.replace('Item B',item_b)        
        # print(text)
        if self.model == "roberta":
            source = self.tokenizer(text, padding='max_length', max_length=self.max_length, return_tensors='pt', truncation=True)
            # source = self.tokenizer.batch_encode_plus([text], 
                # padding='max_length', max_length=self.max_length, return_tensors='pt', truncation=True)
        elif self.model == "kgbert":
            source = self.tokenizer.batch_encode_plus([text], 
                padding='max_length', max_length=self.max_length, return_tensors='pt', truncation=True)
        elif self.model == "gpt2":
            source = self.tokenizer.batch_encode_plus([text +' [EOS]'], padding='max_length', max_length=self.max_length, return_tensors='pt', truncation=True)
        elif self.model == "comet_gpt2":
            if not self.is_eval:
                source = self.tokenizer.batch_encode_plus([text + ' [EOS]'], padding='max_length', max_length=self.max_length, return_tensors='pt', truncation=True)
            else:
                self.tokenizer.padding_side = "left"
                self.tokenizer.pad_token = self.tokenizer.eos_token
                source = self.tokenizer.batch_encode_plus([text + ' [GEN]'], padding='max_length', max_length=self.max_length, return_tensors='pt', truncation=True)

        # if index < 5:
        #     logger.info("Source: {}".format(self.tokenizer.batch_decode(source['input_ids'])))

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        # print(source_ids)
        # print(torch.tensor(self.quality[index]).to(dtype=torch.float))
        lable = None
        
        if float(self.quality[index]) < 0.3:
            label = 0
        else:
            label = 1
        return {
            'ids': source_ids.to(dtype=torch.long),
            'mask': source_mask.to(dtype=torch.long), 
            'label': torch.tensor(label).to(dtype=torch.float),
        }
