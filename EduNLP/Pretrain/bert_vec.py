import os
import json
from xmlrpc.client import Boolean
from EduNLP import logger
from typing import List, Optional, Union, Tuple
from transformers import BertForMaskedLM
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers.file_utils import TensorType
from transformers import BertTokenizer as HFBertTokenizer
from copy import deepcopy
from ..Tokenizer import get_tokenizer
from ..ModelZoo.utils import pad_sequence, load_items
from ..SIF import EDU_SPYMBOLS
from ..ModelZoo.bert import BertForPropertyPrediction
from .pretrian_utils import EduDataset

__all__ = ["BertTokenizer", "BertDataset", "finetune_bert", "finetune_bert_for_property_prediction",
           "finetune_bert_for_knowledge_prediction"]

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


class EduTokenizerForBert(HFBertTokenizer):
    def tokenize(self, text, **kwargs) -> List[str]:
        return self._tokenize(text)

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer._tokenize(text):
                if token in self.all_special_tokens:
                    # If the token is part of the never_split set
                    split_tokens.append(token)
                elif token.encode('utf-8').isalpha():
                    # If token is all English word (Please note that 'textord' will not recognize as new)
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
                else:
                    # If chinese \ punctuation\ other-mixed-english (Please note that '[xxx]' and 'mathod_x' will be recognize as new) 
                    split_tokens.append(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)

        return split_tokens

    def _set_bert_basic_tokenizer(self, tokenize_method, **argv):
        self.basic_tokenizer = get_tokenizer(tokenize_method, **argv)


class BertTokenizer(object):
    """
    Parameters
    ----------
    pretrained_model:
        used pretrained model
    add_specials:
        Whether to add tokens like [FIGURE], [TAG], etc.
    tokenize_method:
        Which text tokenizer to use.
        Must be consistent with TOKENIZER dictionary.

    Returns
    ----------

    Examples
    ----------
    >>> tokenizer = BertTokenizer(add_special_tokens=True)
    >>> item = "有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"
    >>> token_item = tokenizer(item)
    >>> print(token_item.input_ids[:10])
    [101, 1062, 2466, 1963, 1745, 21129, 166, 117, 167, 5276]
    >>> print(tokenizer.tokenize(item)[:10])
    ['公', '式', '如', '图', '[FIGURE]', 'x', ',', 'y', '约', '束']
    >>> items = [item, item]
    >>> token_items = tokenizer(items, return_tensors='pt')
    >>> print(token_items.input_ids.shape)
    torch.Size([2, 27])
    >>> print(len(tokenizer.tokenize(items)))
    2
    >>> tokenizer.save_pretrained('test_dir') # doctest: +SKIP
    >>> tokenizer = BertTokenizer.from_pretrained('test_dir') # doctest: +SKIP
    """

    def __init__(self, pretrained_model="bert-base-chinese", max_length=512, add_specials: (list, bool) = False,
                 tokenize_method=None, **argv):
        self.tokenize_method = tokenize_method
        self.max_length = max_length
        if tokenize_method is not None:
            # In order to be more general for Huggingface's other models,
            # may be we need to inherit and rewrite `_tokenize` for XXTokenizer(PreTrainedTokenizer)
            self.bert_tokenizer = EduTokenizerForBert.from_pretrained(pretrained_model, use_fast=False)
            self._set_basic_tokenizer(tokenize_method, **argv)
        else:
            self.bert_tokenizer = HFBertTokenizer.from_pretrained(pretrained_model, use_fast=False)
        if isinstance(add_specials, bool):
            add_specials = EDU_SPYMBOLS if add_specials else None
        if add_specials is not None:
            self.add_specials(add_specials)

        config = {k: v for k, v in locals().items() if k not in ["self", "__class__", "pretrained_model", "argv"]}
        config.update(argv)
        self.config = config

    def __call__(self, items: Tuple[list, str, dict], key=lambda x: x, padding=True,
                 return_tensors: Optional[Tuple[str, TensorType, bool]] = True, **kwargs):
        if isinstance(items, list):
            text = [key(i) for i in items]
        else:
            text = key(items)
        if isinstance(return_tensors, bool):
            return_tensors = "pt" if return_tensors is True else None
        encodes = self.bert_tokenizer(text, truncation=True, padding=padding, max_length=self.max_length,
                                      return_tensors=return_tensors)
        return encodes

    def __len__(self):
        return len(self.bert_tokenizer)

    def _set_basic_tokenizer(self, tokenize_method: str, **argv):
        self.tokenize_method = tokenize_method
        self.bert_tokenizer._set_bert_basic_tokenizer(tokenize_method, **argv)

    def tokenize(self, items: Union[list, str, dict], key=lambda x: x, **kwargs):
        if isinstance(items, list):
            texts = [self._tokenize(key(i)) for i in items]
            return texts
        else:
            return self._tokenize(key(items))

    def encode(self, items: Tuple[list, str, dict], key=lambda x: x):
        if isinstance(items, str) or isinstance(items, dict):
            return self.bert_tokenizer.encode(key(items))
        else:
            return [self.bert_tokenizer.encode(key(item)) for item in items]

    def decode(self, items: Tuple[list, str, dict], key=lambda x: x):
        if isinstance(items, str) or isinstance(items, dict):
            return self.bert_tokenizer.decode(key(items))
        else:
            return [self.bert_tokenizer.decode(key(item)) for item in items]

    def _tokenize(self, item: Union[str, dict], key=lambda x: x, **kwargs):
        return self.bert_tokenizer._tokenize(key(item))

    @classmethod
    def from_pretrained(cls, tokenizer_config_dir, **argv):
        custom_config_dir = os.path.join(tokenizer_config_dir, 'custom_config.json')
        if os.path.exists(custom_config_dir):
            with open(custom_config_dir, 'r') as f:
                custom_config = json.load(f)
                custom_config.update(argv)
            return cls(tokenizer_config_dir, **custom_config)
        else:
            return cls(tokenizer_config_dir, **argv)

    def save_pretrained(self, tokenizer_config_dir):
        self.bert_tokenizer.save_pretrained(tokenizer_config_dir)
        custom_config = self.config
        with open(os.path.join(tokenizer_config_dir, 'custom_config.json'), 'w') as f:
            json.dump(custom_config, f, indent=2)

    @property
    def vocab_size(self):
        return len(self.bert_tokenizer)

    def set_vocab(self, items: list, key=lambda x: x, lower=False, trim_min_count=1, do_tokenize=True):
        """
        Parameters
        -----------
        items: list
            can be the list of str, or list of dict
        key: function
            determine how to get the text of each item
        """
        word2cnt = dict()
        for item in items:
            tokens = self._tokenize(key(item)) if do_tokenize else key(item)
            if not tokens:
                continue
            for word in tokens:
                word = word.lower() if lower else word
                word2cnt[word] = word2cnt.get(word, 0) + 1
        words = [w for w, c in word2cnt.items() if c >= trim_min_count]
        valid_added_num = self.add_tokens(words)
        return words, valid_added_num

    def add_specials(self, added_spectials: List[str]):
        return self.bert_tokenizer.add_special_tokens({'additional_special_tokens': added_spectials})

    def add_tokens(self, added_tokens: List[str]):
        return self.bert_tokenizer.add_tokens(added_tokens)


class BertDataset(EduDataset):
    pass


def finetune_bert(items: Union[List[dict], List[str]], output_dir: str, pretrained_model="bert-base-chinese",
                  tokenizer_params=None, data_params=None, model_params=None, train_params=None):
    """
    Parameters
    ----------
    items: list, required
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

    Examples
    ----------
    >>> stems = ["有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$",
    ... "有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$"]
    >>> finetune_bert(stems, "examples/test_model/data/data/bert") # doctest: +SKIP
    {'train_runtime': ..., ..., 'epoch': 1.0}
    """
    tokenizer_params = tokenizer_params if tokenizer_params else {}
    data_params = data_params if data_params is not None else {}
    model_params = model_params if model_params is not None else {}
    train_params = train_params if train_params is not None else {}
    # tokenizer configuration
    if os.path.exists(pretrained_model):
        tokenizer = BertTokenizer.from_pretrained(pretrained_model, **tokenizer_params)
    else:
        work_tokenizer_params = {
            "add_specials": True,
            "tokenize_method": "pure_text",
        }
        work_tokenizer_params.update()
        tokenizer = BertTokenizer(pretrained_model, **work_tokenizer_params)
        # TODO: tokenizer.set_vocab()
    # model configuration
    model = BertForMaskedLM.from_pretrained(pretrained_model, **model_params)
    # resize embedding for additional special tokens
    model.resize_token_embeddings(len(tokenizer.bert_tokenizer))

    # dataset configuration
    dataset = BertDataset(tokenizer, items=items,
                          stem_key=data_params.get("stem_key", None))
    mlm_probability = train_params.pop('mlm_probability', 0.15)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer.bert_tokenizer, mlm=True, mlm_probability=mlm_probability
    )
    # training configuration
    work_train_params = deepcopy(DEFAULT_TRAIN_PARAMS)
    work_train_params["output_dir"] = output_dir
    if train_params is not None:
        work_train_params.update(train_params if train_params else {})
    train_args = TrainingArguments(**work_train_params)
    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def finetune_bert_for_property_prediction(train_items,
                                          output_dir,
                                          pretrained_model="bert-base-chinese",
                                          eval_items=None,
                                          tokenizer_params=None,
                                          data_params=None,
                                          train_params=None,
                                          model_params=None
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
    tokenizer = BertTokenizer.from_pretrained(pretrained_model, **tokenizer_params)
    # dataset configuration
    train_dataset = BertDataset(tokenizer=tokenizer, items=train_items,
                                stem_key=data_params.get("stem_key", "ques_content"),
                                label_key=data_params.get("label_key", "difficulty"))
    if eval_items is not None:
        eval_dataset = BertDataset(tokenizer=tokenizer, items=eval_items,
                                   stem_key=data_params.get("stem_key", "ques_content"),
                                   label_key=data_params.get("label_key", "difficulty"))
    else:
        eval_dataset = None
    # model configuration
    model = BertForPropertyPrediction(pretrained_model, **model_params)
    model.bert.resize_token_embeddings(len(tokenizer.bert_tokenizer))
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


def finetune_bert_for_knowledge_prediction(train_items,
                                           output_dir,
                                           pretrained_model="bert-base-chinese",
                                           eval_items=None,
                                           tokenizer_params=None,
                                           data_params=None,
                                           train_params=None,
                                           model_params=None
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
    tokenizer = BertTokenizer.from_pretrained(pretrained_model, **tokenizer_params)
    # dataset configuration
    train_dataset = BertDataset(tokenizer=tokenizer, items=train_items,
                                stem_key=data_params.get("stem_key", "ques_content"),
                                label_key=data_params.get("label_key", "know_list"))
    if eval_items is not None:
        eval_dataset = BertDataset(tokenizer=tokenizer, items=eval_items,
                                   stem_key=data_params.get("stem_key", "ques_content"),
                                   label_key=data_params.get("label_key", "know_list"))
    else:
        eval_dataset = None
    # model configuration
    model = BertForPropertyPrediction(pretrained_model, **model_params)
    model.bert.resize_token_embeddings(len(tokenizer.bert_tokenizer))
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
