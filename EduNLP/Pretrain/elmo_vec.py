import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as tud
import torch.optim as optim
import numpy as np
import json
import os
from EduNLP.Pretrain import BertTokenizer
# from EduNLP.Vector import ElmoModel
from EduNLP.SIF import Symbol, FORMULA_SYMBOL, FIGURE_SYMBOL, QUES_MARK_SYMBOL, TAG_SYMBOL, SEP_SYMBOL
from EduNLP.Tokenizer import PureTextTokenizer

UNK_SYMBOL = '[UNK]'
PAD_SYMBOL = '[PAD]'


class ElmoVocab(object):
    """

    Examples
    --------
    >>> vocab=ElmoVocab()
    >>> items = ["有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"]
    >>> vocab.tokenize(items[0])
    ['公式', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[SEP]', 'z', '=', 'x', '+', '7', 'y', '最大值', '[MARK]']
    >>> len(vocab)
    18
    """

    def __init__(self, path=None):
        # self.bert_tokenizer = BertTokenizer(pretrain_model='bert-base-chinese')
        self.pure_tokenizer = PureTextTokenizer()
        self.t2id = {PAD_SYMBOL: 0, UNK_SYMBOL: 1, FORMULA_SYMBOL: 2, FIGURE_SYMBOL: 3,
                     QUES_MARK_SYMBOL: 4, TAG_SYMBOL: 5, SEP_SYMBOL: 6}
        if path is None:
            pass
        else:
            self.load_vocab(path)

    def __call__(self, item):
        return self.toIndex(item)

    def __len__(self):
        return len(self.t2id)

    def tokenize(self, item: str):
        tokens = next(self.pure_tokenizer([item]))
        for token in tokens:
            self.append(token)
        return tokens

    def toIndex(self, item, max_length=128, pad_to_max_length=True):
        ret = []
        # if len(item) > 0:
        #     if isinstance(item[0], str):
        ret = [self.t2id[UNK_SYMBOL] if token not in self.t2id else self.t2id[token] for token in item]
        if pad_to_max_length:
            if len(ret) < max_length:
                ret = ret + (max_length - len(ret)) * [self.t2id[PAD_SYMBOL]]
            else:
                ret = ret[0:max_length - 1]
        # if isinstance(item[0], list):
        #     ret = [[UNK_SYMBOL if token not in self.t2id else self.t2id[token]
        #             for token in token_list] for token_list in item]
        return ret

    def append(self, item):
        if item in self.t2id:
            pass
        else:
            self.t2id[item] = len(self.t2id)

    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump(self.t2id, f)
        return path

    def load_vocab(self, path):
        with open(path, 'r') as f:
            self.t2id = json.load(f)
        return path


class ElmoDataset(tud.Dataset):
    def __init__(self, texts: list, vocab: ElmoVocab, max_length=128):
        super(ElmoDataset, self).__init__()
        self.vocab = vocab
        self.texts = [text if len(text) < max_length else text[0:max_length - 1] for text in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        sample = {
            'length': len(text),
            'ids': self.vocab.toIndex(text, pad_to_max_length=True)
        }
        return sample


def elmo_collate_fn(batch_data):
    # batch_data = [torch.tensor(t).cuda() for t in batch_data]
    # batch_data = torch.nn.utils.rnn.pad_sequence(batch_data)
    mask = []
    for data in batch_data:
        mask.append([True] * data['length'] + [False] * (len(data['ids']) - data['length']))
    ret_batch = {
        'mask': torch.tensor(mask),
        'length': torch.tensor([data['length'] for data in batch_data]),
        'ids': torch.tensor([data['ids'] for data in batch_data])
    }
    return ret_batch
