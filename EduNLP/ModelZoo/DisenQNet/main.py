# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from DisenQNet import DisenQNet, ConceptModel
from dataset import QuestionDataset
from EduNLP.Pretrain.disenQNet_vec import DisenTokenizer


def parse_args():
    parser = ArgumentParser("DisenQNet")
    # runtime args
    parser.add_argument("--dataset", type=str, dest="dataset", default="data/math23k")
    parser.add_argument("--cuda", type=str, dest="cuda", default=None)
    parser.add_argument("--seed", type=int, dest="seed", default=0)
    parser.add_argument("--log", type=str, dest="log", default=None)

    # model args
    parser.add_argument("--hidden", type=int, dest="hidden", default=128)
    parser.add_argument("--dropout", type=float, dest="dropout", default=0.2)
    parser.add_argument("--cp", type=float, dest="cp", default=1.5)
    parser.add_argument("--mi", type=float, dest="mi", default=1.0)
    parser.add_argument("--dis", type=float, dest="dis", default=2.0)
    parser.add_argument("--pos-weight", type=float, dest="pos_weight", default=1)

    # training args
    parser.add_argument("--epoch", type=int, dest="epoch", default=50)
    parser.add_argument("--batch", type=int, dest="batch", default=128)
    parser.add_argument("--lr", type=float, dest="lr", default=1e-3)
    parser.add_argument("--step", type=int, dest="step", default=20)
    parser.add_argument("--gamma", type=float, dest="gamma", default=0.5)

    # eval args
    parser.add_argument("--vi", action="store_true", dest="vi", default=False)
    parser.add_argument("--topk", type=int, dest="topk", default=2)
    parser.add_argument("--reduction", type=str, dest="reduction", default="micro")

    # dataset args
    parser.add_argument("--trim-min", type=int, dest="trim_min", default=50)
    parser.add_argument("--max-len", type=int, dest="max_len", default=250)

    # adversarial training args
    parser.add_argument("--adv", type=int, dest="adv", default=10)
    parser.add_argument("--warm-up", type=int, dest="warm_up", default=5)

    args = parser.parse_args()
    return args

def init():
    args = parse_args()
    # cuda
    if args.cuda is not None:
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        args.device = "cuda"
    else:
        args.device = "cpu"
    
    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # log
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", filename=args.log)
    return args

def main(args):
    # dataset
    train_path = os.path.join(args.dataset, "train_small.json")
    test_path = os.path.join(args.dataset, "test.json")
    wv_path = os.path.join(args.dataset, "wv.th")
    word_path = os.path.join(args.dataset, "vocab.list")
    concept_path = os.path.join(args.dataset, "concept.list")
    train_dataset = QuestionDataset(train_path, wv_path, word_path, concept_path, args.hidden, args.trim_min, args.max_len, "train", silent=False)
    test_dataset = QuestionDataset(test_path, wv_path, word_path, concept_path, args.hidden, args.trim_min, args.max_len, "test", silent=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=train_dataset.collate_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=test_dataset.collate_data)

    # model
    vocab_size = train_dataset.vocab_size
    concept_size = train_dataset.concept_size
    wv = train_dataset.word2vec

    disen_q_net = DisenQNet(vocab_size, concept_size, args.hidden, args.dropout, args.pos_weight, args.cp, args.mi, args.dis, wv)
    # concept_model = ConceptModel(concept_size, disen_q_net.disen_q_net, args.dropout, args.pos_weight)

    # train and test
    # disen_q_net.train(train_dataloader, test_dataloader, args.device, args.epoch, args.lr, args.step, args.gamma, args.warm_up, args.adv, silent=False)
    # concept_model.train(train_dataloader, test_dataloader, args.device, args.epoch, args.lr, args.step, args.gamma, silent=False, use_vi=args.vi, top_k=args.topk, reduction=args.reduction)
    # disen_q_net.save("disen_q_net.th")
    # concept_model.save("concept_model.th")
    
    disen_q_net.load("disen_q_net.th")
    tokenizer = DisenTokenizer(vocab_path=os.path.join(args.dataset, "vocab.list"))
    test_items = [
        "10 米 的 (2/5) = 多少 米 的 (1/2),有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$",
        "10 米 的 (2/5) = 多少 米 的 (1/2),有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,若$x,y$满足约束条件公式"
    ]
    print("test_items : ", test_items)
    items = tokenizer(test_items)
    embed, k_hidden, i_hidden = disen_q_net.predict(items,device="cuda")

    print(f"embed:{embed.shape}, k_hidden:{k_hidden.shape}, i_hidden:{i_hidden.shape}")
    
    # concept_model.load("concept_model.th")
    return

if __name__ == "__main__":
    args = init()
    main(args)
