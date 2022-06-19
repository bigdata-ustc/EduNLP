import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as tud
import torch.optim as optim
import numpy as np
import json
import os
import time
from EduNLP.SIF import Symbol, FORMULA_SYMBOL, FIGURE_SYMBOL, QUES_MARK_SYMBOL, TAG_SYMBOL, SEP_SYMBOL
from EduNLP.Tokenizer import PureTextTokenizer
from EduNLP.ModelZoo.rnn import ElmoLM, ElmoLMForPreTraining, ElmoLMForDifficultyPrediction
from EduNLP.ModelZoo import set_device
from transformers import TrainingArguments, Trainer

UNK_SYMBOL = '[UNK]'
PAD_SYMBOL = '[PAD]'


class ElmoTokenizer(object):
    """
    Examples
    --------
    >>> t=ElmoTokenizer()
    >>> items = ["有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"]
    >>> t.tokenize(items[0])
    ['公式', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[SEP]', 'z', '=', 'x', '+', '7', 'y', '最大值', '[MARK]']
    >>> len(t)
    18
    """

    def __init__(self, path: str = None):
        """
        Parameters
        ----------
        path: str, optional
            the path of saved ElmoTokenizer, e.g. "../elmo_pub_math/vocab.json"
        """
        self.pure_tokenizer = PureTextTokenizer()
        self.t2id = {PAD_SYMBOL: 0, UNK_SYMBOL: 1, FORMULA_SYMBOL: 2, FIGURE_SYMBOL: 3,
                     QUES_MARK_SYMBOL: 4, TAG_SYMBOL: 5, SEP_SYMBOL: 6}
        if path is None:
            pass
        else:
            self.load_vocab(path)

    def __call__(self, item: (str, list), freeze_vocab=False, pad_to_max_length=False, *args, **kwargs):
        tokens, lengths = self.tokenize(item=item, freeze_vocab=freeze_vocab, return_length=True)
        if isinstance(item, str):
            return self.to_index(item=tokens, pad_to_max_length=pad_to_max_length), lengths
        else:
            ret = []
            for ts in tokens:
                ret.append(self.to_index(item=ts, pad_to_max_length=pad_to_max_length))
            return ret, lengths

    def __len__(self):
        return len(self.t2id)

    def tokenize(self, item: (str, list), freeze_vocab=False, return_length=False):
        items = [item] if isinstance(item, str) else item
        lengths = []
        tokens = []
        for i in self.pure_tokenizer(items):
            tokens.append(i)
            lengths.append(len(i))
            if not freeze_vocab:
                for t in i:
                    self.append(t)
        tokens = tokens[0] if isinstance(item, str) else tokens
        lengths = lengths[0] if isinstance(item, str) else lengths
        if return_length:
            return tokens, lengths
        else:
            return tokens

    def to_index(self, item: list, max_length=128, pad_to_max_length=False):
        ret = [self.t2id[UNK_SYMBOL] if token not in self.t2id else self.t2id[token] for token in item]
        if pad_to_max_length:
            if len(ret) < max_length:
                ret = ret + (max_length - len(ret)) * [self.t2id[PAD_SYMBOL]]
            else:
                ret = ret[0:max_length - 1]
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
    def __init__(self, texts: list, tokenizer: ElmoTokenizer, max_length=128):
        """
        Parameters
        ----------
        texts: list
        tokenizer: ElmoTokenizer
        max_length: int, optional, default=128
        """
        super(ElmoDataset, self).__init__()
        self.tokenizer = tokenizer
        self.texts = [text if len(text) < max_length else text[0:max_length - 1] for text in texts]
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        sample = {
            'length': len(text),
            'idx': self.tokenizer.to_index(text, pad_to_max_length=True, max_length=self.max_length)
        }
        return sample


def elmo_collate_fn(batch_data):
    pred_mask = []
    idx_mask = []
    max_len = max([data['length'] for data in batch_data])
    for data in batch_data:
        pred_mask.append([True] * data['length'] + [False] * (max_len - data['length']))
    for data in batch_data:
        idx_mask.append([True] * data['length'] + [False] * (len(data['idx']) - data['length']))
    ret_batch = {
        'pred_mask': torch.tensor(pred_mask),
        'idx_mask': torch.tensor(idx_mask),
        'seq_len': torch.tensor([data['length'] for data in batch_data]),
        'seq_idx': torch.tensor([data['idx'] for data in batch_data])
    }
    return ret_batch


def train_elmo(texts: list, output_dir: str, pretrained_dir: str = None, emb_dim=512, hid_dim=512, train_params=None):
    """
    Parameters
    ----------
    texts: list, required
        The training corpus of shape (text_num, token_num), a text must be tokenized into tokens
    output_dir: str, required
        The directory to save trained model files
    pretrained_dir: str, optional
        The pretrained model files' directory
    emb_dim: int, optional, default=512
        The embedding dim
    hid_dim: int, optional, default=1024
        The hidden dim
    train_params: dict, optional, default=None
        the training parameters passed to Trainer

    Returns
    -------
    output_dir: str
        The directory that trained model files are saved
    """
    tokenizer = ElmoTokenizer()
    if pretrained_dir:
        tokenizer.load_vocab(os.path.join(pretrained_dir, 'vocab.json'))
    else:
        for text in texts:
            for token in text:
                tokenizer.append(token)
    train_dataset = ElmoDataset(texts, tokenizer)

    if pretrained_dir:
        model = ElmoLMForPreTraining.from_pretrained(pretrained_dir)
    else:
        model = ElmoLMForPreTraining(vocab_size=len(tokenizer), embedding_dim=emb_dim, hidden_size=hid_dim, batch_first=True)

    model.elmo.LM_layer.rnn.flatten_parameters()

    # training parameters
    if train_params:
        epochs = train_params['epochs'] if 'epochs' in train_params else 1
        batch_size = train_params['batch_size'] if 'batch_size' in train_params else 64
        save_steps = train_params['save_steps'] if 'save_steps' in train_params else 100
        save_total_limit = train_params['save_total_limit'] if 'save_total_limit' in train_params else 2
        logging_steps = train_params['logging_steps'] if 'logging_steps' in train_params else 5
        gradient_accumulation_steps = train_params['gradient_accumulation_steps'] \
            if 'gradient_accumulation_steps' in train_params else 1
        learning_rate = train_params['learning_rate'] if 'learning_rate' in train_params else 5e-4
    else:
        # default
        epochs = 1
        batch_size = 64
        save_steps = 1000
        save_total_limit = 2
        logging_steps = 5
        gradient_accumulation_steps = 1
        learning_rate = 5e-4

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=elmo_collate_fn,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_vocab(os.path.join(output_dir, 'vocab.json'))
    return output_dir



def 

def train_elmo_for_difficulty_prediction(texts: list, output_dir: str, pretrained_dir: str = None, emb_dim=512,
                                         hid_dim=512, train_params=None):
    tokenizer = ElmoTokenizer()
    if pretrained_dir:
        tokenizer.load_vocab(os.path.join(pretrained_dir, 'vocab.json'))
    else:
        for text in texts:
            for token in text:
                tokenizer.append(token)
    train_dataset = ElmoDataset(texts, tokenizer)

    if pretrained_dir:
        model = ElmoLMForDifficultyPrediction.from_pretrained(pretrained_dir)
    else:
        model = ElmoLMForDifficultyPrediction(vocab_size=len(tokenizer), embedding_dim=emb_dim, hidden_size=hid_dim, batch_first=True)

    model.elmo.LM_layer.rnn.flatten_parameters()



    # training parameters
    if train_params:
        epochs = train_params['epochs'] if 'epochs' in train_params else 1
        batch_size = train_params['batch_size'] if 'batch_size' in train_params else 64
        save_steps = train_params['save_steps'] if 'save_steps' in train_params else 100
        save_total_limit = train_params['save_total_limit'] if 'save_total_limit' in train_params else 2
        logging_steps = train_params['logging_steps'] if 'logging_steps' in train_params else 5
        gradient_accumulation_steps = train_params['gradient_accumulation_steps'] \
            if 'gradient_accumulation_steps' in train_params else 1
        learning_rate = train_params['learning_rate'] if 'learning_rate' in train_params else 5e-4
    else:
        # default
        epochs = 1
        batch_size = 64
        save_steps = 1000
        save_total_limit = 2
        logging_steps = 5
        gradient_accumulation_steps = 1
        learning_rate = 5e-4

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=elmo_collate_fn,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_vocab(os.path.join(output_dir, 'vocab.json'))
    return output_dir