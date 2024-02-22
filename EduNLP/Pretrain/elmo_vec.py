import torch
import os
from copy import deepcopy
from transformers import TrainingArguments, Trainer
from typing import Optional, Union, List
from ..ModelZoo.rnn import ElmoLM, ElmoLMForPreTraining, ElmoLMForPropertyPrediction, ElmoLMForKnowledgePrediction
from ..ModelZoo.utils import pad_sequence
from .pretrian_utils import PretrainedEduTokenizer, EduDataset
from ..utils import logger

__all__ = ["ElmoTokenizer", "ElmoDataset", "pretrain_elmo", "pretrain_elmo_for_property_prediction",
           "pretrain_elmo_for_knowledge_prediction"]

DEFAULT_TRAIN_PARAMS = {
    # default
    "output_dir": None,
    "overwrite_output_dir": True,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 32,
    # evaluation_strategy: "steps",
    # eval_steps:200,
    "save_steps": 1000,
    "save_total_limit": 2,
    # "load_best_model_at_end": False,
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
    >>> len(t)
    14
    >>> t.tokenize(items[0])
    ['公式', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[SEP]', 'z', '=', 'x', '+', '7', 'y', '最大值', '[MARK]']
    >>> t(items[0])
    {'seq_idx': tensor([1, 1, 6, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1, 7]), 'seq_len': tensor(17)}
    >>> t.set_vocab(items[0])
    ['公式', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[SEP]', 'z', '=', 'x', '+', '7', 'y', '最大值', '[MARK]']
    >>> len(t)
    45
    >>> t(items[0])
    {'seq_idx': tensor([ 1,  1,  6, 26, 27, 28,  1,  1,  9, 35, 36, 26, 37, 38, 28,  1,  7]), 'seq_len': tensor(17)}
    """

    def __init__(self, vocab_path=None, max_length=250, tokenize_method="pure_text", add_specials=True, **kwargs):
        super().__init__(vocab_path=vocab_path, max_length=max_length, tokenize_method=tokenize_method,
                         add_specials=add_specials, **kwargs)


class ElmoDataset(EduDataset):
    def __init__(self, tokenizer: ElmoTokenizer, **kwargs):
        super(ElmoDataset, self).__init__(tokenizer=tokenizer, **kwargs)

    def collate_fn(self, batch_data):
        pad_idx = self.tokenizer.vocab.pad_idx
        first = batch_data[0]
        batch = {
            k: [item[k] for item in batch_data] for k in first.keys()
        }
        batch["seq_idx"] = pad_sequence(batch["seq_idx"], pad_val=pad_idx)
        batch = {key: torch.as_tensor(val) for key, val in batch.items()}
        return batch


def pretrain_elmo(train_items: Union[List[dict], List[str]] = None, output_dir: str = None, pretrained_dir: str = None,
                  tokenizer_params=None, tokenizer=None, data_params=None, model_params=None, train_params=None):
    """
    Parameters
    ----------
    train_items: list, required
        The training corpus, each item could be str or dict
    output_dir: str, required
        The directory to save trained model files
    pretrained_dir: str, optional
        The pretrained directory for model and tokenizer
    tokenizer_params: dict, optional, default=None
        The parameters passed to ElmoTokenizer
    data_params: dict, optional, default=None
        - stem_key
        - label_key
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
    if tokenizer is None:
        if pretrained_dir is not None and os.path.exists(pretrained_dir):
            tokenizer = ElmoTokenizer.from_pretrained(pretrained_dir, **tokenizer_params)
        else:
            work_tokenizer_params = {
                "add_specials": True,
                "tokenize_method": "pure_text",
            }
            work_tokenizer_params.update(tokenizer_params)
            tokenizer = ElmoTokenizer(**work_tokenizer_params)
            corpus_items = train_items
            if isinstance(corpus_items[0], str):
                tokenizer.set_vocab(corpus_items, trim_min_count=data_params.get('trim_min_count', 2))
            else:
                tokenizer.set_vocab(corpus_items,
                                    key=lambda x: x[data_params.get("stem_key", "ques_content")],
                                    trim_min_count=data_params.get('trim_min_count', 2))
    logger.info("prepare ElmoDataset")
    # dataset configuration
    dataset = ElmoDataset(tokenizer=tokenizer, items=train_items,
                          stem_key=data_params.get("stem_key", "ques_content"))
    logger.info("prepare ElmoLMForPreTraining")
    # model configuration
    if pretrained_dir:
        model = ElmoLMForPreTraining.from_pretrained(pretrained_dir, **model_params)
    else:
        work_model_params = {
            "vocab_size": len(tokenizer),
            "embedding_dim": 300,
            "hidden_size": 300
        }
        work_model_params.update(model_params if model_params else {})
        model = ElmoLMForPreTraining(**work_model_params)
    model.elmo.LM_layer.rnn.flatten_parameters()

    logger.info("train start!")
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
    # trainer.model.save_pretrained(output_dir)
    assert isinstance(trainer.model, ElmoLMForPreTraining)
    trainer.save_model(output_dir)
    trainer.model.save_config(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def pretrain_elmo_for_property_prediction(
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
    tokenizer_params = tokenizer_params if tokenizer_params else {}
    data_params = data_params if data_params is not None else {}
    model_params = model_params if model_params is not None else {}
    train_params = train_params if train_params is not None else {}
    # tokenizer configuration
    if pretrained_dir is not None:
        tokenizer = ElmoTokenizer.from_pretrained(pretrained_dir, **tokenizer_params)
    else:
        work_tokenizer_params = {
            "add_special_tokens": True,
            "tokenize_method": "pure_text",
        }
        work_tokenizer_params.update(tokenizer_params if tokenizer_params else {})
        tokenizer = ElmoTokenizer(**work_tokenizer_params)
        corpus_items = train_items + eval_items if eval_items else []
        tokenizer.set_vocab(corpus_items,
                            key=lambda x: x[data_params.get("stem_key", "ques_content")])
    # dataset configuration
    train_dataset = ElmoDataset(tokenizer=tokenizer, items=train_items,
                                stem_key=data_params.get("stem_key", "ques_content"),
                                label_key=data_params.get("label_key", "difficulty"))
    if eval_items is not None:
        eval_dataset = ElmoDataset(tokenizer=tokenizer, items=eval_items,
                                   stem_key=data_params.get("stem_key", "ques_content"),
                                   label_key=data_params.get("label_key", "difficulty"))
    else:
        eval_dataset = None
    # model configuration
    if pretrained_dir is not None:
        model = ElmoLMForPropertyPrediction.from_pretrained(pretrained_dir, **model_params)
    else:
        work_model_params = {
            "vocab_size": len(tokenizer),
            "embedding_dim": 512,
            "hidden_size": 512
        }
        work_model_params.update(model_params if model_params else {})
        model = ElmoLMForPropertyPrediction(**work_model_params)
    model.elmo.LM_layer.rnn.flatten_parameters()

    logger.info("train start!")
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
        data_collator=train_dataset.collate_fn,
    )
    trainer.train()
    # trainer.model.save_pretrained(output_dir)
    assert isinstance(trainer.model, ElmoLMForPropertyPrediction)
    trainer.save_model(output_dir)
    trainer.model.save_config(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def pretrain_elmo_for_knowledge_prediction(
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
    tokenizer_params = tokenizer_params if tokenizer_params else {}
    data_params = data_params if data_params is not None else {}
    model_params = model_params if model_params is not None else {}
    train_params = train_params if train_params is not None else {}
    # tokenizer configuration
    if pretrained_dir is not None:
        tokenizer = ElmoTokenizer.from_pretrained(pretrained_dir, **tokenizer_params)
    else:
        work_tokenizer_params = {
            "add_special_tokens": True,
            "tokenize_method": "pure_text",
        }
        work_tokenizer_params.update(tokenizer_params if tokenizer_params else {})
        tokenizer = ElmoTokenizer(**work_tokenizer_params)
        corpus_items = train_items + eval_items if eval_items else []
        tokenizer.set_vocab(corpus_items,
                            key=lambda x: x[data_params.get("stem_key", "ques_content")])
    # dataset configuration
    train_dataset = ElmoDataset(tokenizer=tokenizer, items=train_items,
                                stem_key=data_params.get("stem_key", "ques_content"),
                                label_key=data_params.get("label_key", "know_list"))
    if eval_items is not None:
        eval_dataset = ElmoDataset(tokenizer=tokenizer, items=eval_items,
                                   stem_key=data_params.get("stem_key", "ques_content"),
                                   label_key=data_params.get("label_key", "know_list"))
    else:
        eval_dataset = None
    # model configuration
    if pretrained_dir is not None:
        model = ElmoLMForKnowledgePrediction.from_pretrained(pretrained_dir, **model_params)
    else:
        work_model_params = {
            "vocab_size": len(tokenizer),
            "embedding_dim": 512,
            "hidden_size": 512
        }
        work_model_params.update(model_params if model_params else {})
        model = ElmoLMForKnowledgePrediction(**work_model_params)
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
        data_collator=train_dataset.collate_fn,
    )
    trainer.train()
    assert isinstance(trainer.model, ElmoLMForKnowledgePrediction)
    trainer.save_model(output_dir)
    trainer.model.save_config(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
