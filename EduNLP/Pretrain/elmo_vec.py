from tkinter.messagebox import NO
from matplotlib.colors import NoNorm
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as tud
import torch.optim as optim
import numpy as np
import json
import os
import time
import multiprocessing
from copy import deepcopy
from typing import Dict, List
from ..SIF import EDU_SPYMBOLS
from ..Tokenizer import PureTextTokenizer
from ..ModelZoo.rnn import ElmoLM, ElmoLMForPreTraining, ElmoLMForPropertyPrediction
from ..ModelZoo.utils import pad_sequence
from .pretrian_utils import PretrainedEduTokenizer, EduDataset
from transformers import TrainingArguments, Trainer, PretrainedConfig
from datasets import load_dataset
from datasets import Dataset as HFDataset
import datasets
import pandas as pd
from typing import Optional, Union, List, Dict

__all__ = ["ElmoTokenizer", "train_elmo", "train_elmo_for_perporty_prediction"]

DEFAULT_TRAIN_PARAMS = {
    # default
    "output_dir": None,
    "overwrite_output_dir": True,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    # evaluation_strategy: "steps",
    # eval_steps:200,
    "save_steps": 1000,
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    # metric_for_best_model: "loss",
    # greater_is_better: False,
    "logging_dir": None,
    "logging_steps": 5,
    "gradient_accumulation_steps": 1,
    "learning_rate": 5e-4,
    # disable_tqdm: True,
    # no_cuda: True,
}


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
        super().__init__(vocab_path=vocab_path, max_length=max_length, tokenize_method=tokenize_method, **argv)


"""Note: Be Make sure Tokenizer output batched tensors by default"""
class ElmoDataset(EduDataset):
    def __init__(self, tokenizer: ElmoTokenizer, **argv):
        super(ElmoDataset, self).__init__(tokenizer=tokenizer, **argv)

    def collate_fn(self, batch_data):
        pad_idx = self.tokenizer.vocab.pad_idx
        first =  batch_data[0]
        batch = {
            k: [item[k] for item in batch_data] for k in first.keys()
        }
        batch["seq_idx"] = pad_sequence(batch["seq_idx"], pad_val=pad_idx, max_length=500)
        batch = {key: torch.as_tensor(val) for key, val in batch.items()}
        return batch


def train_elmo(items: Union[List[dict], List[str]], output_dir: str, pretrain_dir: str = None,
               tokenizer_params=None, data_params=None, model_params=None, train_params=None):
    """
    Parameters
    ----------
    items: list, required
        The training corpus, each item could be str or dict
    output_dir: str, required
        The directory to save trained model files
    pretrained_dir: str, optional
        The pretrained directory for model and tokenizer
    tokenizer_params: dict, optional, default=None
        The parameters passed to ElmoTokenizer
    data_params: dict, optional, default=None
        The parameters passed to ElmoDataset and ElmoTokenizer
    model_params: dict, optional, default=None
        The parameters passed to Trainer
    train_params: dict, optional, default=None
    """
    # tokenizer configuration
    if os.path.exists(pretrain_dir):
        tokenizer = ElmoTokenizer.from_pretrained(pretrain_dir)
    else:
        work_tokenizer_params = {
            "add_special_tokens": True,
            "text_tokenizer": "pure_text",
        }
        work_tokenizer_params.update(tokenizer_params if tokenizer_params else {})
        tokenizer = ElmoTokenizer(pretrain_dir, **work_tokenizer_params)
        corpus_items = items
        if isinstance(items[0], str):
            tokenizer.set_vocab(corpus_items)
        else:
            tokenizer.set_vocab(corpus_items,
                                key=lambda x: x[data_params.get("feature_key", "stem")])

    # dataset configuration
    dataset = ElmoDataset(items=items, tokenizer=tokenizer,
                          feature_key=data_params.get("feature_key", None))

    # model configuration
    if pretrain_dir:
        model = ElmoLMForPreTraining.from_pretrained(pretrain_dir)
    else:
        work_model_params = {
            "vocab_size": len(tokenizer),
            "embedding_dim": 512,
            "hidden_size": 512
        }
        work_model_params.update(model_params if model_params else {})
        model = ElmoLMForPreTraining(**work_model_params)
    model.elmo.LM_layer.rnn.flatten_parameters()

    # training configuration
    work_train_params = deepcopy(DEFAULT_TRAIN_PARAMS)
    work_train_params["output_dir"] = output_dir
    if train_params is not None:
        work_train_params.update(train_params if train_params else {})
    work_args = TrainingArguments(**work_train_params)
    trainer = Trainer(
        model=model,
        args=work_args,
        train_dataset=dataset,
        data_collator=dataset.collate_fn,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def train_elmo_for_perporty_prediction(
        train_items: list, output_dir: str, pretrained_dir=None, eval_items=None,
        tokenizer_params=None, data_params=None, train_params=None, model_params=None
    ):
    """
    Parameters
    ----------
    train_items: list, required
        The training items, each item could be str or dict
    output_dir: str, required
        The directory to save trained model files
    pretrained_dir: str, optional
        The pretrained directory for model and tokenizer
    eval_items: list, required
        The evaluating items, each item could be str or dict
    tokenizer_params: dict, optional, default=None
        The parameters passed to ElmoTokenizer
    data_params: dict, optional, default=None
        The parameters passed to ElmoDataset and ElmoTokenizer
    model_params: dict, optional, default=None
        The parameters passed to Trainer
    train_params: dict, optional, default=None
    """
    # tokenizer configuration
    if pretrained_dir is not None:
        tokenizer = ElmoTokenizer.from_pretrained(pretrained_dir)
    else:
        work_tokenizer_params = {
            "add_special_tokens": True,
            "text_tokenizer": "pure_text",
        }
        work_tokenizer_params.update(tokenizer_params if tokenizer_params else {})
        tokenizer = ElmoTokenizer(pretrained_dir, **work_tokenizer_params)
        corpus_items = train_items + eval_items
        tokenizer.set_vocab(corpus_items,
                            key=lambda x: x[data_params.get("feature_key", "stem")])
    # dataset configuration
    train_dataset = EduDataset(items=train_items, tokenizer=tokenizer,
                               feature_key=data_params.get("feature_key", "stem"),
                               labal_key=data_params.get("labal_key", "diff"))
    if eval_items is not None:
        eval_dataset = EduDataset(items=eval_items, tokenizer=tokenizer,
                                  feature_key=data_params.get("feature_key", "stem"),
                                  labal_key=data_params.get("labal_key", "diff"))

    # model configuration
    if pretrained_dir is not None:
        model = ElmoLMForPropertyPrediction.from_pretrained(pretrained_dir)
    else:
        work_model_params = {
            "vocab_size": len(tokenizer),
            "embedding_dim": 512,
            "hidden_size": 512
        }
        work_model_params.update(model_params if model_params else {})
        model = ElmoLMForPropertyPrediction(**work_model_params)
    model.elmo.LM_layer.rnn.flatten_parameters()

    # training configuration
    work_train_params = deepcopy(DEFAULT_TRAIN_PARAMS)
    work_train_params["output_dir"] = output_dir
    if train_params is not None:
        work_train_params.update(train_params if train_params else {})
    work_train_params = TrainingArguments(**work_train_params)  
    trainer = Trainer(
        model=model,
        args=work_train_params,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ElmoDataset.elmo_collate_fn,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir