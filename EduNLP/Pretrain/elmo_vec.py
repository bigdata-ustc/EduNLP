import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as tud
import torch.optim as optim
import numpy as np
import json
import os
import time
from typing import Dict, List
from ..SIF import EDU_SPYMBOLS
from ..Tokenizer import PureTextTokenizer
from ..ModelZoo.rnn import ElmoLM, ElmoLMForPreTraining, ElmoLMForPropertyPrediction
from ..ModelZoo.utils import pad_sequence
from .pretrian_utils import PretrainedEduTokenizer
from transformers import TrainingArguments, Trainer, PretrainedConfig
from datasets import load_dataset
from datasets import Dataset as HFDataset
import datasets
import pandas as pd

UNK_SYMBOL = '[UNK]'
PAD_SYMBOL = '[PAD]'


class ElmoTokenizer(PretrainedEduTokenizer):
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

    def __init__(self, vocab_path=None, max_length=250, tokenize_method="pure_text", **argv):
        super().__init__(vocab_path, max_length, tokenize_method, **argv)


class ElmoDataset(tud.Dataset):
    def __init__(self, texts: List[str], tokenizer: ElmoTokenizer):
        """
        Parameters
        ----------
        texts: list
        tokenizer: ElmoTokenizer
        max_length: int, optional, default=128
        """
        super(ElmoDataset, self).__init__()

        self.tokenizer = tokenizer
        self.items = texts
        ds = HFDataset.from_dict({"text": texts})
        ds = ds.map(self._preprocess, batched=True, batch_size=1000)
        self.ds = ds.remove_columns("text")

    def _preprocess(self, sample):
        return self.tokenizer(sample["text"], padding=False, return_tensors=False, return_text=False)

    def __len__(self):
        return self.ds.num_rows

    def __getitem__(self, index):
        return self.ds[index]

    def collate_fn(self, batch_data):
        pad_idx = self.tokenizer.vocab.pad_idx
        first = batch_data[0]
        batch = {
            k: [item[k] for item in batch_data] for k in first.keys()
        }
        batch["seq_idx"] = pad_sequence(batch["seq_idx"], pad_val=pad_idx, max_length=500)
        batch = {key: torch.as_tensor(val) for key, val in batch.items()}
        return batch


class ElmoForPPDataset(tud.Dataset):
    def __init__(self, items, tokenizer, mode="train", feature_key="content", labal_key="difficulty"):
        self.tokenizer = tokenizer
        self.items = items
        self.mode = mode
        self.feature_key = feature_key
        self.labal_key = labal_key

        if mode in ["train", "val"]:
            columns = [feature_key, labal_key]
        else:
            columns = [feature_key]
        ds = HFDataset.from_pandas(pd.DataFrame(items)[columns])
        ds = ds.map(self._preprocess, batched=True, batch_size=1000)
        ds = ds.remove_columns(feature_key)
        self.ds = ds.rename_columns({labal_key: "labels"})

    def _preprocess(self, sample):
        return self.tokenizer(sample[self.feature_key], padding=False, return_tensors=False, return_text=False)

    def __len__(self):
        return self.ds.num_rows

    def __getitem__(self, index):
        return self.ds[index]

    def collate_fn(self, batch_data):
        pad_idx = self.tokenizer.vocab.pad_idx
        first = batch_data[0]
        batch = {
            k: [item[k] for item in batch_data] for k in first.keys()
        }
        batch["seq_idx"] = pad_sequence(batch["seq_idx"], pad_val=pad_idx, max_length=500)
        batch = {key: torch.as_tensor(val) for key, val in batch.items()}
        return batch


def train_elmo(texts: list, output_dir: str, pretrained_dir: str = None, emb_dim=512, hid_dim=512, train_params=None):
    """
    Parameters
    ----------
    texts: list, required
        The training corpus of shape (text_num)
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
        tokenizer = ElmoTokenizer.from_pretrained(pretrained_dir)
    else:
        tokenizer.set_vocab(texts)
    train_dataset = ElmoDataset(texts, tokenizer)

    if pretrained_dir:
        model = ElmoLMForPreTraining.from_pretrained(pretrained_dir)
    else:
        model = ElmoLMForPreTraining(
            vocab_size=len(tokenizer),
            embedding_dim=emb_dim,
            hidden_size=hid_dim,
            batch_first=True
        )

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
        data_collator=train_dataset.collate_fn,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def train_elmo_for_difficulty_prediction(texts: list, output_dir: str, pretrained_dir: str = None, emb_dim=512,
                                         hid_dim=512, train_params=None):
    tokenizer = ElmoTokenizer()
    if pretrained_dir:
        tokenizer = ElmoTokenizer.from_pretrained(pretrained_dir)
    else:
        tokenizer.set_vocab(texts)
    train_dataset = ElmoDataset(texts, tokenizer)

    if pretrained_dir:
        model = ElmoLMForPropertyPrediction.from_pretrained(pretrained_dir)
    else:
        model = ElmoLMForPropertyPrediction(vocab_size=len(tokenizer), embedding_dim=emb_dim, hidden_size=hid_dim,
                                            batch_first=True)

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
        data_collator=ElmoDataset.elmo_collate_fn,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
