import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
import torch
import time
import random
import argparse
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# from models.pseudo_labeling.model import RoBERTAClassifier
from model import RoBERTAClassifier
from model_utils import evaluate
from dataloader import CKBPDataset
from transformers import AutoTokenizer,RobertaTokenizer
from torch.utils.data import random_split

# from utils.ckbp_utils import special_token_list


logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()


    # model related
    group_model = parser.add_argument_group("model configs")
    group_model.add_argument("--ptlm", default='roberta-base', type=str, 
                        required=False, help="choose from huggingface PTLM")
    group_model.add_argument("--pretrain_from_path", required=False, default="",
                    help="pretrain this model from a checkpoint") # a bit different from --ptlm.


    # pseudo-label-related args
    group_pseudo_label = parser.add_argument_group("pseudo label configs")
    group_pseudo_label.add_argument("--pseudo_examples_path", required=False,
                        help="paths to the csv file containing pseudo-labels")
    group_pseudo_label.add_argument("--pseudo_proportion_in_batch", required=False, default=0.1, type=float,
                        help="proportion of pseudo labeled data in a batch when mixing labeled and pseudo labeled data")

    # training-related args
    group_trainer = parser.add_argument_group("training configs")

    group_trainer.add_argument("--device", default='cuda', type=str, required=False,
                    help="device")
    group_trainer.add_argument("--optimizer", default='ADAM', type=str, required=False,
                    help="optimizer")
    group_trainer.add_argument("--lr", default=0.01, type=float, required=False,
                    help="learning rate")
    group_trainer.add_argument("--lrdecay", default=1, type=float, required=False,
                        help="learning rate decay every x steps")
    group_trainer.add_argument("--decay_every", default=500, type=int, required=False,
                    help="show test result every x steps")
    group_trainer.add_argument("--batch_size", default=32, type=int, required=False,
                        help="batch size")
    group_trainer.add_argument("--test_batch_size", default=32, type=int, required=False,
                        help="test batch size")
    group_trainer.add_argument("--epochs", default=3, type=int, required=False,
                        help="batch size")
    group_trainer.add_argument("--steps", default=-1, type=int, required=False,
                        help="the number of iterations to train model on labeled data. used for the case training model less than 1 epoch")
    group_trainer.add_argument("--max_length", default=100, type=int, required=False,
                        help="max_seq_length of h+r+t")
    group_trainer.add_argument("--eval_metric", type=str, required=False, default="overall_auc",
                    choices=["grouped_auc", "overall_auc", "accuracy"],
                    help="evaluation metric.")
    group_trainer.add_argument("--eval_every", default=5, type=int, required=False,
                        help="eval on test set every x steps.")
    group_trainer.add_argument("--relation_as_special_token", action="store_true",
                        help="whether to use special token to represent relation.")
    

    # IO-related

    group_data = parser.add_argument_group("IO related configs")
    group_data.add_argument("--output_dir", default="results",
                        type=str, required=False,
                        help="where to output.")
    group_data.add_argument("--train_csv_path", default='', type=str, required=True)
    group_data.add_argument("--evaluation_file_path", default="/home/ubuntu/project/data/elec_true_all_v1.1.csv", 
                            type=str, required=False)
    group_data.add_argument("--model_dir", default='models', type=str, required=False,
                        help="Where to save models.") # TODO
    group_data.add_argument("--save_best_model", action="store_true",
                        help="whether to save the best model.")
    group_data.add_argument("--log_dir", default='logs', type=str, required=False,
                        help="Where to save logs.") #TODO
    group_data.add_argument("--experiment_name", default='', type=str, required=False,
                        help="A special name that will be prepended to the dir name of the output.") # TODO
    
    group_data.add_argument("--seed", default=401, type=int, required=False,
                    help="random seed")

    args = parser.parse_args()

    return args

def main():


    # get all arguments
    args = parse_args()

    save_dir = './result2'
    os.makedirs(save_dir, exist_ok=True)

    # logger = logging.getLogger("kg-bert")
    # handler = logging.FileHandler(os.path.join(save_dir, f"log_seed_{args.seed}.txt"))
    # logger.addHandler(handler)

    # set random seeds
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # load model
    model = RoBERTAClassifier(args.ptlm).to(args.device)

    # tokenizer = AutoTokenizer.from_pretrained(args.ptlm)
    tokenizer = RobertaTokenizer.from_pretrained(args.ptlm)
    sep_token = tokenizer.sep_token

    # if args.relation_as_special_token:
    #     tokenizer.add_special_tokens({
    #         'additional_special_tokens': special_token_list,
    #     })
    #     model.model.resize_token_embeddings(len(tokenizer))


    # load data

    train_dataset = pd.read_csv(args.train_csv_path)
    test_dataset = pd.read_csv(args.evaluation_file_path)
    # print(train_dataset.columns)
    train_params = {
        'batch_size': int(args.batch_size*(1-args.pseudo_proportion_in_batch)),
        'shuffle': True,
        'num_workers': 5
    }

    val_params = {
        'batch_size': args.test_batch_size,
        'shuffle': False,
        'num_workers': 5
    }

    training_set = CKBPDataset(train_dataset, tokenizer, args.max_length, sep_token=sep_token)
    train_l = int(0.6 * len(train_dataset))
    pseudo_l = int(0.2 * len(train_dataset))
    test_l = int(0.1 * len(train_dataset))
    dev_l = len(train_dataset) - train_l - pseudo_l - test_l
    train_dataset_, pseudo_dataset_ ,test_dataset_, dev_dataset_ = random_split(
        dataset=training_set,
        lengths=[train_l, pseudo_l, test_l, dev_l],
        generator=torch.Generator().manual_seed(0)
    )
    train = DataLoader(train_dataset_, **train_params, drop_last=True)
    pseudo = DataLoader(pseudo_dataset_, **train_params, drop_last=True)
    test = DataLoader(test_dataset_, **train_params, drop_last=True)
    dev = DataLoader(dev_dataset_, **train_params, drop_last=True)
    # print(len(train))
    # print(len(pseudo))
    # print(len(test))

    training_loader = DataLoader(training_set, **train_params, drop_last=True)

    # train = training_set.sample(frac=0.8).reset_index(drop=True)
    # pseudo = training_set.drop(train.index).reset_index(drop=True)

    batch_count = len(train)

    dev_dataset = CKBPDataset(test_dataset, tokenizer, args.max_length, sep_token=sep_token) 
    tst_dataset = CKBPDataset(test_dataset, tokenizer, args.max_length, sep_token=sep_token) 
    # dev_dataset = CKBPDataset(train_dataset, tokenizer, args.max_length, sep_token=sep_token) 
    # tst_dataset = CKBPDataset(train_dataset, tokenizer, args.max_length, sep_token=sep_token) 

    dev_dataloader = DataLoader(dev_dataset, **val_params, drop_last=False)
    tst_dataloader = DataLoader(tst_dataset, **val_params, drop_last=False)


    # pseudo label dataset
    pseudo_params = {
        'batch_size': args.batch_size - train_params["batch_size"],
        'shuffle': True,
        'num_workers': 5
    }

    pseudo_dataset = pd.read_csv(args.pseudo_examples_path)
    # pseudo_dataset = pd.read_csv(args.train_csv_path)
    pseudo_dataset = pseudo_dataset.sample(n=pseudo_params['batch_size']*batch_count, random_state=args.seed, replace=True).reset_index(drop=True)

    pseudo_training_set = CKBPDataset(pseudo_dataset, tokenizer, args.max_length, sep_token=sep_token)
    pseudo_loader = DataLoader(pseudo_training_set, **pseudo_params, drop_last=True)

    # model training
    
    if args.optimizer == "ADAM":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    criterion = torch.nn.CrossEntropyLoss()

    best_epoch, best_iter = 0, 0
    best_val_score = 0.0

    model.train()
    
    if args.steps > 0:
        args.epochs = 1

    for e in range(args.epochs):
        print('epochs')
        print(e)
        for iteration, (ldata, pdata) in tqdm(enumerate(zip(train, pseudo), 1), total=batch_count):
            # the iteration starts from 1. 
            y = torch.cat((ldata['label'], pdata['label']), dim=0).to(args.device, dtype=torch.long)
            ids = torch.cat((ldata['ids'], pdata['ids']), dim=0).to(args.device, dtype=torch.long)
            mask = torch.cat((ldata['mask'], pdata['mask']), dim=0).to(args.device, dtype=torch.long)

            tokens = {"input_ids":ids, "attention_mask":mask}
            logits = model(tokens)
            print(y.min())
            print(y.max())
            # print(y)
            # print(logits)
            y[y<0] = 0
            y[y>1] = 1
            logits[logits<0] = 0
            logits[logits>1] = 1
            print(logits.min())
            print(logits.max())
            loss = criterion(logits, y)
            # print('loss')
            # print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.eval_every > 0 and iteration % args.eval_every == 0:
                model.eval()

                # validation auc
                # eval_auc = (evaluate(tokenizer, model, args.device, test)).item()
                eval_auc, _ = evaluate(tokenizer, model, args.device, test)
                print('eval_auc: ')
                # assert _ == len(dev_dataset)
                eval_auc = eval_auc.item()
                print(eval_auc)
                if float(eval_auc) > best_val_score:
                    best_val_score = eval_auc
                    if args.save_best_model:
                        torch.save(model.state_dict(), save_dir + f"/best_model_seed_{args.seed}.pth")
                        tokenizer.save_pretrained(save_dir + "/best_tokenizer")
                    
                    best_epoch, best_iter = e, iteration
                    print('kkkkkkkkkkkkkkkkk')
                    print(f"Best validation score reached at epoch {best_epoch} step {best_iter}")

                # calc test scores after every x step
                # tst_auc = (evaluate(tokenizer, model, args.device, dev, class_break_down=True)).item()
                tst_auc, _ = evaluate(tokenizer, model, args.device, dev)
                tst_auc = tst_auc.item()

                print(f"Overall auc & Test Set & CSKB Head + ASER tail & ASER edges")
                print('scroes: '+ str(tst_auc))
                # print(f"test scores at epoch {e} step {iteration}:" + " & ".join([str(round(tst_auc*100, 1))]+\
                #         [str(round(class_scores[clss]*100, 1)) for clss in ["test_set", "cs_head", "all_head"]]) )

                model.train()
            
            if args.steps > 0 and iteration >= args.steps:
                break

if __name__ == "__main__":
    main()
