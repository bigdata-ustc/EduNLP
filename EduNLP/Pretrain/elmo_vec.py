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
from EduNLP.Vector import ElmoModel

UNK_SYMBOL = '[UNK]'
PAD_SYMBOL = '[PAD]'


class ElmoVocab(object):
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

    def tokenize(self, item):
        tokens = self.pure_tokenizer.tokenize(item)
        if len(tokens) > 0:
            if isinstance(tokens[0], str):
                for token in tokens:
                    self.append(token)
            if isinstance(tokens[0], list):
                for token_list in tokens:
                    for token in token_list:
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
    """
    # Parameters

    texts: 'list'
        The corpus, consist of strings
    vocab: 'ElmoVocab'
    """

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


def finetune_elmo(token_items, vocab: ElmoVocab, pretrain_model_weights: str, output_path: str,
                  dataloader_params: dict = None, epochs: int = 1):
    """

    Parameters
    ----------
    token_items: list of torch.tensor, (required)
        A list containing tensors.
    vocab: ElmoVocab, (required)
        The ELMo vocabulary
    pretrain_model_weights: str, (required)
        The path of pretrained ELMo network weights¬
    output_path: str, (required)
        The path to save fine-tuned network weights.
    epochs: int, default = 1, (optional)
        Training epochs
    dataloader_params: dict(optional)
        Pytorch DataLoader parameters

    Examples
    ----------
    >>> elmo_vocab=ElmoVocab()
    >>> elmo_vocab.load_vocab('examples/test_model/data/elmo/vocab_wiki.json')
    'examples/test_model/data/elmo/vocab_wiki.json'
    >>> stems = ["有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$",
    ... "有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$"]
    >>> items = [elmo_vocab.toIndex(doc) for doc in stems]
    >>> weights_dir = 'examples/test_model/data/elmo/elmo_pretrain_weights_wiki.bin'
    >>> saving_dir = 'examples/test_model/data/elmo/elmo_finetune_weights_wiki.bin'
    >>> finetune_elmo(items, elmo_vocab, weights_dir, saving_dir)
    Epoch: 0
    """
    token_items = [torch.tensor([token_id if token_id < len(vocab) else 0 for token_id in tensor]) for tensor in
                   token_items]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elmo = ElmoModel(t2id=vocab.t2id, lr=1e-5)
    elmo.load_weights(pretrain_model_weights)
    training_set = ElmoDataset(token_items)
    if dataloader_params is None:
        dataloader_params = {'shuffle': False, 'num_workers': 0}
    else:
        pass
    training_generator = tud.DataLoader(training_set, **dataloader_params, collate_fn=collate_fn)
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        for batch in training_generator:
            batch = batch.to(device)
            elmo.train(batch)
    elmo.save_weights(output_path)


def train_elmo(texts, filepath_prefix: str, pretrain_model: str, emb_dim=512, hid_dim=1024, batch_size=4,
               epochs=1):
    vocab = ElmoVocab()
    if pretrain_model is None:
        texts = vocab.tokenize(texts)  # This WILL append new token to vocabulary
        vocab.save_vocab(filepath_prefix + '/' + os.path.basename(filepath_prefix) + '_elmo_vocab.json')
    else:
        vocab.load_vocab(filepath_prefix + '/' + pretrain_model + '_elmo_vocab.json')
        texts = vocab.pure_tokenizer.tokenize(texts)  # This will NOT append new token to vocabulary
    dataset = ElmoDataset(texts=texts, vocab=vocab)
    model = ElmoModel(elmo_vocab=vocab, emb_size=emb_dim, hidden_size=hid_dim)
    model.train(train_set=dataset, batch_size=batch_size, epochs=epochs)
    if pretrain_model is None:
        model.save_weights(filepath_prefix + '/' + os.path.basename(filepath_prefix) + '_elmo_weights.bin')
    else:
        model.save_weights(filepath_prefix + '/' + pretrain_model + '_elmo_weights.bin')
