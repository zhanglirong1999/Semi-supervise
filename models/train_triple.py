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
from torch.utils.data import random_split
# from pytorch_lightning import LightningModule, Trainer
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from model import RoBERTAClassifier
from model_utils import evaluate
from dataloader_tuple import TupleDataset
from model_triple import RoBERTATripletModel
from transformers import AutoTokenizer,RobertaTokenizer

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()


    # model related
    group_model = parser.add_argument_group("model configs")
    group_model.add_argument("--ptlm", default='roberta-base', type=str, 
                        required=False, help="choose from huggingface PTLM")
    group_model.add_argument("--pretrain_from_path", required=False, default="",
                    help="pretrain this model from a checkpoint") # a bit different from --ptlm.


    #### TODO: pseudo-label-related args
    group_pseudo_label = parser.add_argument_group("pseudo label configs")

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
    group_trainer.add_argument("--epochs", default=10, type=int, required=False,
                        help="batch size")
    group_trainer.add_argument("--steps", default=-1, type=int, required=False,
                        help="the number of iterations to train model on labeled data. used for the case training model less than 1 epoch")
    group_trainer.add_argument("--max_length", default=30, type=int, required=False,
                        help="max_seq_length of h+r+t")
    group_trainer.add_argument("--eval_metric", type=str, required=False, default="overall_auc",
                    choices=["grouped_auc", "overall_auc", "accuracy"],
                    help="evaluation metric.")
    group_trainer.add_argument("--eval_every", default=100, type=int, required=False,
                        help="eval on test set every x steps.")
    group_trainer.add_argument("--relation_as_special_token", action="store_true",
                        help="whether to use special token to represent relation.")
    group_trainer.add_argument("--noisy_training", action="store_true",
                        help="whether to have a noisy training, flip the labels with probability p_noisy.")
    group_trainer.add_argument("--p_noisy", default=0.0, type=float, required=False,
                    help="probability to flip the labels")

    # IO-related

    group_data = parser.add_argument_group("IO related configs")
    group_data.add_argument("--output_dir", default="triple_results",
                        type=str, required=False,
                        help="where to output.")
    group_data.add_argument("--train_csv_path", default='/data/wwangbw/Lirong/data/three_triple.csv', type=str, required=False)
    group_data.add_argument("--evaluation_file_path", default="/data/wwangbw/Lirong/data/three_triple.csv", 
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

    experiment_name = args.experiment_name
    if args.noisy_training:
        experiment_name = experiment_name + f"_noisy_{args.p_noisy}"

    save_dir = os.path.join(args.output_dir, "_".join([os.path.basename(args.ptlm), 
        f"bs{args.batch_size}", f"evalstep{args.eval_every}"])+experiment_name )
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("kg-bert")
    handler = logging.FileHandler(os.path.join(save_dir, f"log_seed_{args.seed}.txt"))
    logger.addHandler(handler)

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
    model = RoBERTATripletModel(args.ptlm).to(args.device)

    tokenizer = RobertaTokenizer.from_pretrained(args.ptlm)

    if "bart" in args.ptlm:
        sep_token = " "
    else:
        sep_token = tokenizer.sep_token

    if args.relation_as_special_token:
        tokenizer.add_special_tokens({
            'additional_special_tokens': special_token_list,
        })
        model.model.resize_token_embeddings(len(tokenizer))


    # load data

    train_dataset = pd.read_csv(args.train_csv_path)
    # infer_file = pd.read_csv(args.evaluation_file_path)
    training_set = TupleDataset(train_dataset, tokenizer, args.max_length, sep_token=sep_token)

    train_l = int(0.8 * len(train_dataset))
    test_l = len(train_dataset) - train_l
    train_dataset_,  test_dataset_= random_split(
        dataset=training_set,
        lengths=[train_l, test_l],
        generator=torch.Generator().manual_seed(0)
    )
    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 5
    }

    val_params = {
        'batch_size': args.test_batch_size,
        'shuffle': True,
        'num_workers': 5
    }
    train = DataLoader(train_dataset_, **train_params, drop_last=True)
    test = DataLoader(test_dataset_, **val_params, drop_last=True)

    # training_set = CKBPDataset(train_dataset_, tokenizer, args.max_length, sep_token=sep_token)
    # training_loader = DataLoader(training_set, **train_params, drop_last=True)

    # tst_dataset = CKBPDataset(test, tokenizer, args.max_length, sep_token=sep_token) 

    # tst_dataloader = DataLoader(tst_dataset, **val_params, drop_last=False)

    # model training
    
    if args.optimizer == "ADAM":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.PairwiseDistance())
    # criterion = torch.nn.BCEWithLogitsLoss()

    best_epoch, best_iter = 0, 0
    best_val_score = 0
    best_f1 = 0

    model.train()
    batch_count = len(train)
    final_loss = 20
    iteration = 0

    for e in range(args.epochs):

        for iteration, data in tqdm(enumerate(train, iteration+1),total=batch_count):
            # the iteration starts from 1. 
            anchor = data['ids_anchor'].to(args.device, dtype=torch.long)
            mask_anchor = data['mask_anchor'].to(args.device, dtype=torch.long)
            ids_positive = data['ids_positive'].to(args.device, dtype=torch.long)
            attention_mask = data['mask_positive'].to(args.device, dtype=torch.long)
            ids_negative = data['ids_negative'].to(args.device, dtype=torch.long)
            mask_negative = data['mask_negative'].to(args.device, dtype=torch.long)
            y_anchor = data['label_anchor'].to(args.device, dtype=torch.long)
            y_positive = data['label_positive'].to(args.device, dtype=torch.long)
            y_negative = data['label_negative'].to(args.device, dtype=torch.long)
            # print(y)
            # noisy training
            if args.noisy_training:
                noisy_vec = torch.rand(len(y))
                y = y ^ (noisy_vec < args.p_noisy).to(args.device)
                # flip label with probability p_noisy

            tokens = {"ids_anchor":anchor, "mask_anchor":mask_anchor,
            "ids_positive":ids_positive, "mask_positive":attention_mask,
            "ids_negative":ids_negative, "mask_negative":mask_negative
            }

            emb_anchor, emb_positive, emb_negative = model(tokens)
            # print(logits)
            # print(y)
            loss = criterion(emb_anchor, emb_positive, emb_negative)
            # print('loss1 %f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('loss')
            print(loss)

            if loss < final_loss:
                torch.save(model.state_dict(), save_dir + f"/best_model_seed_{args.seed}.pth")
                tokenizer.save_pretrained(save_dir + "/best_tokenizer")
                
    print('best loss %f' % (final_loss))

if __name__ == "__main__":
    main()
