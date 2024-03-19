import os
from typing import List, Union
from transformers import BertForMaskedLM
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from copy import deepcopy

from ..ModelZoo.jiuzhang import JiuzhangForKnowledgePrediction, JiuzhangForPropertyPrediction
from .pretrian_utils import EduDataset
from .hugginface_utils import TokenizerForHuggingface

__all__ = [
    "JiuzhangTokenizer",
    "JiuzhangDataset",
    "finetune_jiuzhang_for_property_prediction",
    "finetune_jiuzhang_for_knowledge_prediction",
]

DEFAULT_TRAIN_PARAMS = {
    # default
    "output_dir": None,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 32,
    # "per_device_eval_batch_size": 32,
    # evaluation_strategy: "steps",
    # eval_steps:200,
    "save_steps": 1000,
    "save_total_limit": 2,
    # "load_best_model_at_end": True,
    # metric_for_best_model: "loss",
    # greater_is_better: False,
    "logging_dir": None,
    "logging_steps": 5,
    "gradient_accumulation_steps": 1,
    "learning_rate": 5e-5,
    # disable_tqdm: True,
    # no_cuda: True,
}


class JiuzhangTokenizer(TokenizerForHuggingface):
    """
    Examples
    ----------
    >>> tokenizer = JiuzhangTokenizer(add_special_tokens=True)
    >>> item = "有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"
    >>> token_item = tokenizer(item)
    >>> print(token_item.input_ids)
    tensor([[ 101, 1062, 2466, 1963, 1745,  138,  100,  140,  166,  117,  167, 5276,
             3338, 3340,  816, 1062, 2466,  102,  168,  134,  166,  116,  128,  167,
             3297, 1920,  966,  138,  100,  140,  102]])
    >>> print(tokenizer.tokenize(item)[:10])
    ['公', '式', '如', '图', '[', '[UNK]', ']', 'x', ',', 'y']
    >>> items = [item, item]
    >>> token_items = tokenizer(items, return_tensors='pt')
    >>> print(token_items.input_ids.shape)
    torch.Size([2, 31])
    >>> print(len(tokenizer.tokenize(items)))
    2
    >>> tokenizer.save_pretrained('test_dir') # doctest: +SKIP
    >>> tokenizer = JiuzhangTokenizer.from_pretrained('test_dir') # doctest: +SKIP
    """

    pass


class JiuzhangDataset(EduDataset):
    pass


def finetune_jiuzhang_for_property_prediction(
    train_items,
    output_dir,
    pretrained_model="bert-base-chinese",
    eval_items=None,
    tokenizer_params=None,
    data_params=None,
    train_params=None,
    model_params=None,
):
    """
    Parameters
    ----------
    train_items: list, required
        The training corpus, each item could be str or dict
    output_dir: str, required
        The directory to save trained model files
    pretrained_model: str, optional
        The pretrained model name or path for model and tokenizer
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
    tokenizer_params = tokenizer_params if tokenizer_params else {}
    data_params = data_params if data_params is not None else {}
    model_params = model_params if model_params is not None else {}
    train_params = train_params if train_params is not None else {}
    # tokenizer configuration
    tokenizer = JiuzhangTokenizer.from_pretrained(pretrained_model, **tokenizer_params)
    # dataset configuration
    train_dataset = JiuzhangDataset(
        tokenizer=tokenizer,
        items=train_items,
        stem_key=data_params.get("stem_key", "ques_content"),
        label_key=data_params.get("label_key", "difficulty"),
    )
    if eval_items is not None:
        eval_dataset = JiuzhangDataset(
            tokenizer=tokenizer,
            items=eval_items,
            stem_key=data_params.get("stem_key", "ques_content"),
            label_key=data_params.get("label_key", "difficulty"),
        )
    else:
        eval_dataset = None
    # model configuration
    model = JiuzhangForPropertyPrediction(pretrained_model, **model_params)
    model.jiuzhang.resize_token_embeddings(len(tokenizer.bert_tokenizer))
    # training configuration
    work_train_params = deepcopy(DEFAULT_TRAIN_PARAMS)
    work_train_params["output_dir"] = output_dir
    if train_params is not None:
        work_train_params.update(train_params if train_params else {})
    train_args = TrainingArguments(**work_train_params)
    data_collator = DataCollatorWithPadding(tokenizer.bert_tokenizer)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    # trainer.model.save_pretrained(output_dir)
    trainer.save_model(output_dir)
    trainer.model.save_config(output_dir)
    tokenizer.save_pretrained(output_dir)


def finetune_jiuzhang_for_knowledge_prediction(
    train_items,
    output_dir,
    pretrained_model="bert-base-chinese",
    eval_items=None,
    tokenizer_params=None,
    data_params=None,
    train_params=None,
    model_params=None,
):
    """
    Parameters
    ----------
    train_items: list, required
        The training corpus, each item could be str or dict
    output_dir: str, required
        The directory to save trained model files
    pretrained_model: str, optional
        The pretrained model name or path for model and tokenizer
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
    tokenizer_params = tokenizer_params if tokenizer_params else {}
    data_params = data_params if data_params is not None else {}
    model_params = model_params if model_params is not None else {}
    train_params = train_params if train_params is not None else {}
    # tokenizer configuration
    tokenizer = JiuzhangTokenizer.from_pretrained(pretrained_model, **tokenizer_params)
    # dataset configuration
    train_dataset = JiuzhangDataset(
        tokenizer=tokenizer,
        items=train_items,
        stem_key=data_params.get("stem_key", "ques_content"),
        label_key=data_params.get("label_key", "know_list"),
    )
    if eval_items is not None:
        eval_dataset = JiuzhangDataset(
            tokenizer=tokenizer,
            items=eval_items,
            stem_key=data_params.get("stem_key", "ques_content"),
            label_key=data_params.get("label_key", "know_list"),
        )
    else:
        eval_dataset = None
    # model configuration
    model = JiuzhangForKnowledgePrediction(
        pretrained_model_dir=pretrained_model, **model_params
    )
    model.jiuzhang.resize_token_embeddings(len(tokenizer.bert_tokenizer))
    # training configuration
    work_train_params = deepcopy(DEFAULT_TRAIN_PARAMS)
    work_train_params["output_dir"] = output_dir
    if train_params is not None:
        work_train_params.update(train_params if train_params else {})
    train_args = TrainingArguments(**work_train_params)
    data_collator = DataCollatorWithPadding(tokenizer.bert_tokenizer)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    # trainer.model.save_pretrained(output_dir)
    trainer.save_model(output_dir)
    trainer.model.save_config(output_dir)
    tokenizer.save_pretrained(output_dir)
