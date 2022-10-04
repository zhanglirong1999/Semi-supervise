from cProfile import label
from readline import append_history_file
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import logging
from dataloader import CKBPDataset
from transformers import AutoTokenizer,RobertaTokenizer
import os
from statistics import mode
import sys
import torch
import time
import random
import argparse
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def main():
    seed =401
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    train_dataset = pd.read_csv('../../data/TOTAL_typicality_result.csv')
    # train_dataset = pd.read_csv('../../data/test.csv')

    print(len(train_dataset))

    positive = train_dataset[train_dataset['quality']>0.6]
    negative = train_dataset[train_dataset['quality']<0.2]
    print(len(positive))
    print(len(negative))

    mid_length = int(len(positive)/2)
    anchor = list()
    pos = list()
    neg = list()
    anchor_label = list()
    pos_label = list()
    neg_label = list()
    # print(positive)
    index = 0
    for _, data in positive.iterrows():
        # print(data)
        if index >= mid_length:
            break
        text_anchor = getSentence(data)
        # print(positive.iloc[index+mid_length])
        text_positive = getSentence(positive.iloc[index+mid_length])
        text_negative = getSentence(negative.iloc[index])
        label_anchor = data['quality']
        label_positive = positive.iloc[index+mid_length]['quality']
        label_negative = negative.iloc[index]['quality']

        anchor.append(text_anchor)
        pos.append(text_positive)
        neg.append(text_negative)
        anchor_label.append(label_anchor)
        pos_label.append(label_positive)
        neg_label.append(label_negative)

        index += 1
    
    data_list = [anchor,anchor_label,pos,pos_label,neg,neg_label]

    df = pd.DataFrame (data_list).transpose()
    df.columns = ['anchor','anchor_label','positive','positive_label','negative','negative_label']
    
    df.to_csv('../../data/three_triple.csv')
    print('successfully saved')


def getSentence(data):
    # print(data)
    item_a = str(data["item_a_name"])
    item_b = str(data["item_b_name"])
    assertion = data["assertion"] 
    # print(type(item_a))
    text = assertion
    text = text.replace('Item A',item_a)
    text = text.replace('Item B',item_b) 
    # print(text)
    return text


if __name__ == "__main__":
    main()
