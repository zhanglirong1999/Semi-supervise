from cProfile import label
from cgi import print_directory
import random
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from pytorch_metrics import BinaryAccuracy

cr_loss = torch.nn.functional.cross_entropy

def evaluate(tokenizer, model, device, loader, class_break_down=False, model_type="kgbert"):
    # evaluate CSKB Population

    model.eval()

    predicted_scores = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    classes = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            # print(data)
            y = data['label'].to(device, dtype=torch.float)
            # print(y)
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)

            tokens = {"input_ids":ids, "attention_mask":mask}

            if model_type == "kgbert" or model_type=="roberta":
                outputs_logits = model(tokens)

                logits = torch.softmax(outputs_logits, dim=1)            
                values = logits[:, 1]
                # print(values)
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
            labels = torch.cat((labels, y))
            classes.extend(data["clss"])
    y_true = labels
    y_pred = predicted_scores
    y_true[y_true>=0.7] = 1
    y_true[y_true<0.7] = 0
    y_pred[y_pred>=0.7] = 1
    y_pred[y_pred<0.7] = 0
    # print(len(labels))
    # print(len(predicted_scores))
    f1 = f1_score(y_true=y_true.cpu(), y_pred=y_pred.cpu())
    print(f1)
    return f1
    metrics = BinaryAccuracy(device).to(device)
    # preds = predicted_scores
    # preds[preds >= 0.7] = 1 
    metrics.update(y_pred, y_true)
    return metrics.compute(), len(labels)
    # return roc_auc_score( y_true.tolist(), (y_pred).tolist()), len(labels)


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
