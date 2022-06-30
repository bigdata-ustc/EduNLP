import os
import json
from EduNLP import logger
from typing import List, Optional, Union
from transformers import BertModelForMaskedLM
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


__all__ = ["BertTokenizer", "finetune_bert", "train_bert_for_perporty_predition"]

DEFAULT_TRAIN_PARAMS = {
    # default
    "output_dir": None,
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
                # If the token is part of the never_split set
                if token in self.all_special_tokens:
                    split_tokens.append(token)
                # If token is all English word 
                # (Please note that '[xxx]' and 'mathod_x' work well, but 'textord' will break down)
                elif token.encode('utf-8').isalpha():
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
                # If chinese \ punctuation\ other-mixed-english
                else:
                    split_tokens.append(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        
        return split_tokens
    
    def set_bert_basic_tokenizer(self, text_tokenizer, **argv):
        self.basic_tokenizer = get_tokenizer[text_tokenizer](**argv)


class BertTokenizer(object):
    """
    Parameters
    ----------
    pretrain_model:
        used pretrained model
    add_special_tokens:
        Whether to add tokens like [FIGURE], [TAG], etc.
    text_tokenizer:
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
    def __init__(self, pretrain_model="bert-base-chinese", add_special_tokens=False, text_tokenizer=None, **argv):
        self.bert_tokenizer = EduTokenizerForBert.from_pretrained(pretrain_model, use_fast=False)
        self.add_special_tokens = add_special_tokens
        if add_special_tokens:
            self.add_specials(EDU_SPYMBOLS)
        self.text_tokenizer_name = text_tokenizer
        if text_tokenizer is not None:
            # In order to be more general for Huggingface's other models,
            # may be we need to inherit and rewrite `_tokenize` for XXTokenizer(PreTrainedTokenizer)
            self.bert_tokenizer.set_bert_basic_tokenizer(text_tokenizer, **argv)

        config = {k: v for k, v in locals().items() if k not in ["self", "__class__", "pretrain_model", "argv"]}
        config.update(argv)
        self.config = config

    def add_specials(self, added_spectials: List[str]):
        self.bert_tokenizer.add_special_tokens({'additional_special_tokens': added_spectials})

    def add_tokens(self, added_tokens: List[str]):
        self.bert_tokenizer.add_tokens(added_tokens)

    def set_vocab(self, items: list, key=lambda x: x, lower=False, trim_min_count=1, tokenize=True):
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
            tokens = self.bert_tokenizer.tokenize(key(item)) if tokenize else key(item)
            for word in tokens:
                word = word.lower() if lower else word
                word2cnt[word] = word2cnt.get(word, 0) + 1
        words = [w for w, c in word2cnt.items() if c >= trim_min_count]
        self.add_tokens(words)
        return words

    def __call__(self, item: Union[list, str], key=lambda x: x, return_tensors: Optional[Union[str, TensorType]] = None,
                 *args, **kwargs):
        if isinstance(item, str):
            item = key(item)
        else:
            item = [key(i) for i in item]
        return self.bert_tokenizer(item, truncation=True, padding=True, return_tensors=return_tensors)

    def __len__(self):
        return len(self.bert_tokenizer)

    def tokenize(self, item: Union[list, str], *args, key=lambda x: x, **kwargs):
        if isinstance(item, str):
            return self.bert_tokenizer.tokenize(key(item))
        else:
            item = [self.bert_tokenizer.tokenize(key(i)) for i in item]
            return item

    def save_pretrained(self, tokenizer_config_dir):
        self.bert_tokenizer.save_pretrained(tokenizer_config_dir)
        custom_config = self.config
        with open(os.path.join(tokenizer_config_dir, 'custom_config.json'), 'w') as f:
            json.dump(custom_config, f, indent=2)

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


class BertDataset(EduDataset):
    pass

def finetune_bert(items: Union[List[dict], List[str]], output_dir: str, pretrain_model="bert-base-chinese",
                  tokenizer_params=None, data_params=None, model_params=None, train_params=None):
    """
    Parameters
    ----------
    items: list, required
        The training corpus, each item could be str or dict
    output_dir: str, required
        The directory to save trained model files
    pretrain_model: str, optional
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
    # tokenizer configuration
    if os.path.exists(pretrain_model):
        tokenizer = BertTokenizer.from_pretrained(pretrain_model)
    else:
        work_tokenizer_params = {
            "add_special_tokens": True,
            "text_tokenizer": "pure_text",
        }
        work_tokenizer_params.update(tokenizer_params if tokenizer_params else {})
        tokenizer = BertTokenizer(pretrain_model, **work_tokenizer_params)
        # todo: tokenizer.set_vocab()
    # model configuration
    model = BertModelForMaskedLM.from_pretrained(pretrain_model, **model_params)
    # resize embedding for additional special tokens
    model.bert.resize_token_embeddings(len(tokenizer.bert_tokenizer))

    # dataset configuration  
    dataset = BertDataset(items=items, tokenizer=tokenizer,
                         feature_key=data_params.get("feature_key", None))
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


def train_bert_for_perporty_predition(train_items, output_dir, pretrain_model="bert-base-chinese",
                                      eval_items=None, 
                                      data_params=None, train_params=None, model_params=None):
    """
    Parameters
    ----------
    items: list, required
        The training corpus, each item could be str or dict
    output_dir: str, required
        The directory to save trained model files
    pretrain_model: str, optional
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
    # tokenizer configuration
    tokenizer = BertTokenizer.from_pretrained(pretrain_model)
    # dataset configuration
    train_dataset = BertDataset(items=train_items, tokenizer=tokenizer,
                               feature_key=data_params.get("feature_key", "stem"),
                               labal_key=data_params.get("labal_key", "diff"))
    if eval_items is not None:
        eval_dataset = BertDataset(items=eval_items, tokenizer=tokenizer,
                                  feature_key=data_params.get("feature_key", "stem"),
                                  labal_key=data_params.get("labal_key", "diff"))
    # model configuration
    model = BertForPropertyPrediction(pretrain_model, **model_params)
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
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
