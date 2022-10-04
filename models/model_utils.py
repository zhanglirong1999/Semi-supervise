from cProfile import label
from cgi import print_directory
import random
from itertools import chain

from urllib3 import Retry

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from pytorch_metrics import BinaryAccuracy
from torchmetrics.functional import stat_scores

cr_loss = torch.nn.functional.cross_entropy

def evaluate(tokenizer, model, device, loader, class_break_down=False, model_type="kgbert"):
    # evaluate CSKB Population

    # model.eval()
    accuracy = float(-1) 
    f1 = float(-1) 
    predicted_scores = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    # with torch.no_grad():
    for iteration, data in enumerate(loader, 0):
            # print(data)
            y = data['label'].to(device, dtype=torch.float)
            # print(y)
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)

            tokens = {"input_ids":ids, "attention_mask":mask}

            # if model_type == "kgbert" or model_type=="roberta":
                # print(tokens)
            # print('k')
            # print(tokens)
            output = model(tokens)
            # print(output)

            # result = model.compute_metric(output, y)
            # print(result)
            # if f1 < result['f1']:
            #     f1 =  result['f1']
            # if accuracy < result['accuracy']:
            #     accuracy =  result['accuracy']
            predict_scores = torch.sigmoid(output)
            # print(predict_scores)
            values = torch.argmax(predict_scores, dim=-1)
            predicted_scores = torch.cat((predicted_scores, values))
            labels = torch.cat((labels, y))
    # return f1, accuracy
    # return f1_score( labels.tolist(), (predicted_scores).tolist())
    return f1_score( labels.tolist(), (predicted_scores).tolist()), accuracy_score( labels.tolist(), (predicted_scores).tolist())
    # return stat_scores(y_pred, y_true, reduce='macro', num_classes=2, multiclass=True)

def score_triples(tokenizer, model, device, loader, model_type="kgbert"):
    """
        return: predicted_scores (list) The scores predicted by the model.
                for KG-BERT, the returned score is the softmax score for the triple being true.
                    GPT2, the returned score is the negative GPT2 loss.
    """
    model.eval()

    predicted_scores = torch.tensor([]).to(device)

    with torch.no_grad():
        for _, data in tqdm(enumerate(loader, 0)):
            y = data['label'].to(device, dtype=torch.long)
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)

            tokens = {"input_ids":ids, "attention_mask":mask}

            if model_type == "kgbert":
                outputs_logits = model(tokens)

                logits = torch.softmax(outputs_logits, dim=1)            
                values = logits[:, 1]
            elif model_type == "gpt2":
                outputs = model(input_ids = ids, attention_mask = mask, labels=ids)

                shift_logits = outputs[1][..., :-1, :].contiguous().view(-1,outputs[1].size(-1))
                shift_labels = ids[..., 1:].contiguous().view(-1)
                
                losses = cr_loss(shift_logits, shift_labels, 
                    ignore_index=tokenizer.pad_token_id, reduction="none").view(ids.size(0), -1)

                losses = torch.div(torch.sum(losses, dim=1), 
                    torch.sum(mask[:, 1:], dim=1)) # (batch_size, ) get the loss after removing PAD_TOKEN

                values = -losses

            predicted_scores = torch.cat((predicted_scores, values))

    return predicted_scores.tolist()
