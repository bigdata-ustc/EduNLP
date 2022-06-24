import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as tud
import torch.optim as optim
import numpy as np
import json
import os
import time

from ..SIF import EDU_SPYMBOLS
from ..Tokenizer import PureTextTokenizer
from ..ModelZoo.rnn import ElmoLM, ElmoLMForPreTraining, ElmoLMForPropertyPrediction
from ..ModelZoo.utils import pad_sequence
from .pretrian_utils import PretrainedTokenizer
from transformers import TrainingArguments, Trainer, PretrainedConfig

UNK_SYMBOL = '[UNK]'
PAD_SYMBOL = '[PAD]'


class ElmoTokenizer(PretrainedTokenizer):
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
    def __init__(self, texts: list, tokenizer: ElmoTokenizer):
        """
        Parameters
        ----------
        texts: list
        tokenizer: ElmoTokenizer
        max_length: int, optional, default=128
        """
        super(ElmoDataset, self).__init__()
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = tokenizer.max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        item = self.texts[index]
        return self.tokenizer(item, padding=False, return_tensor=False, return_text=False)

    def collate_fn(self, batch_data):
        pad_idx = self.tokenizer.vocab.pad_idx
        first =  batch_data[0]
        batch = {
            k: [item[k] for item in batch_data] for k in first.keys()
        }
        batch["seq_idx"] = pad_sequence(batch["seq_idx"], pad_val=pad_idx)

        # old版本使用max_len进行pad
        pred_mask, idx_mask = [], []
        max_len, batch_len = max(batch["seq_len"]), batch["seq_idx"].shape[1]
        for data in batch_data:
            pred_mask.append([True] * data['seq_len'] + [False] * (max_len - data['seq_len']))
            idx_mask.append([True] * data['seq_len'] + [False] * (batch_len - data['seq_len']))
        batch["pred_mask"] = pred_mask
        batch["idx_mask"] = idx_mask

        batch = {key: torch.as_tensor(val) for key, val in batch.items()}
        return batch


class ElmoForPPDataset(tud.Dataset):
    pass


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
        model = ElmoLMForPropertyPrediction(vocab_size=len(tokenizer), embedding_dim=emb_dim, hidden_size=hid_dim, batch_first=True)

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