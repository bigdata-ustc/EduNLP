import os
import json
from EduNLP import logger
from typing import List, Optional, Union
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers.file_utils import TensorType
from transformers import BertTokenizer as HFBertTokenizer
from transformers import PretrainedConfig
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
import pandas as pd
from ..Tokenizer import PureTextTokenizer, TOKENIZER
from ..ModelZoo.utils import pad_sequence, load_items
from ..SIF import Symbol, EDU_SPYMBOLS
from ..ModelZoo.bert import BertForPropertyPrediction


__all__ = ["BertTokenizer", "finetune_bert"]


class EduTokenizerForBert(HFBertTokenizer):
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
        assert text_tokenizer in TOKENIZER, f"text_tokenizer should be one of {list(TOKENIZER.keys())}"
        self.basic_tokenizer = TOKENIZER[text_tokenizer](**argv)


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
            # customize_tokens = []
            # for i in EDU_SPYMBOLS:
            #     if i not in self.bert_tokenizer.additional_special_tokens:
            #         customize_tokens.append(Symbol(i))
            # if customize_tokens:
            #     self.bert_tokenizer.add_special_tokens({'additional_special_tokens': customize_tokens})

        self.text_tokenizer_name = text_tokenizer
        if text_tokenizer is not None:
            # In order to be more general for Huggingface's other models,
            # may be we need to inherit and rewrite `_tokenize` for XXTokenizer(PreTrainedTokenizer)
            self.bert_tokenizer.set_bert_basic_tokenizer(text_tokenizer, **argv)

        config = {k: v for k, v in locals().items() if k not in ["self", "__class__", "pretrain_model"]}
        print("[debug] config: ", config)
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
            tokens = self.bert_tokenizer._tokenize(key(item)) if tokenize else key(item)
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
            return cls(tokenizer_config_dir)


class FinetuneDataset(Dataset):
    def __init__(self, items):
        self.items = items
        self.len = len(items)

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return self.len


class BertForPPDataset(Dataset):
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
        self.ds = HFDataset.from_pandas(pd.DataFrame(items)[columns])
        self.ds = self.ds.map(lambda sample: tokenizer(sample[feature_key]), batched=True, batch_size=1000)
        self.ds = self.ds.remove_columns(feature_key)
        if mode in ["train", "val"]:
            self.ds = self.ds.rename_columns({labal_key: "labels"})

    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self):
        return len(self.items)


def finetune_bert(items, output_dir, pretrain_model="bert-base-chinese", train_params=None):
    """

    Parameters
    ----------
    items：dict
        the tokenization results of questions
    output_dir: str
        the path to save the model
    pretrain_model: str
        the name or path of pre-trained model
    train_params: dict
        the training parameters passed to Trainer

    Examples
    ----------
    >>> tokenizer = BertTokenizer()
    >>> stems = ["有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$",
    ... "有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$"]
    >>> token_item = [tokenizer(i) for i in stems]
    >>> print(token_item[0].keys())
    dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
    >>> finetune_bert(token_item, "examples/test_model/data/data/bert") # doctest: +SKIP
    {'train_runtime': ..., ..., 'epoch': 1.0}
    """
    model = AutoModelForMaskedLM.from_pretrained(pretrain_model)
    tokenizer = BertTokenizer(pretrain_model, add_special_tokens=True)
    # resize embedding for additional special tokens
    model.resize_token_embeddings(len(tokenizer.bert_tokenizer))

    # training parameters
    if train_params:
        mlm_probability = train_params['mlm_probability'] if 'mlm_probability' in train_params else 0.15
        epochs = train_params['epochs'] if 'epochs' in train_params else 1
        batch_size = train_params['batch_size'] if 'batch_size' in train_params else 64
        save_steps = train_params['save_steps'] if 'save_steps' in train_params else 100
        save_total_limit = train_params['save_total_limit'] if 'save_total_limit' in train_params else 2
        logging_steps = train_params['logging_steps'] if 'logging_steps' in train_params else 5
        gradient_accumulation_steps = train_params['gradient_accumulation_steps'] \
            if 'gradient_accumulation_steps' in train_params else 1
    else:
        # default
        mlm_probability = 0.15
        epochs = 1
        batch_size = 64
        save_steps = 1000
        save_total_limit = 2
        logging_steps = 5
        gradient_accumulation_steps = 1

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer.bert_tokenizer, mlm=True, mlm_probability=mlm_probability
    )

    dataset = FinetuneDataset(items)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer.bert_tokenizer,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def train_bert_for_perporty_predition(train_items, output_dir, pretrain_model="bert-base-chinese",
                                      eval_items=None, train_params=None, model_params=None):
    model = BertForPropertyPrediction(pretrain_model, **model_params)
    tokenizer = BertTokenizer(pretrain_model, add_special_tokens=True)
    
    train_dataset = BertForPPDataset(train_items, tokenizer)
    if eval_items is not None:
        eval_dataset = BertForPPDataset(eval_items, tokenizer)


    if train_params:
        epochs = train_params.pop('epochs') if 'epochs' in train_params else 1
        batch_size = train_params.pop('batch_size') if 'batch_size' in train_params else 64
        
        save_steps = train_params.pop('save_steps') if 'save_steps' in train_params else 100
        save_total_limit = train_params.pop('save_total_limit') if 'save_total_limit' in train_params else 2
        logging_dir= train_params.pop('logging_dir') if 'logging_dir' in train_params else None
        logging_steps = train_params.pop('logging_steps') if 'logging_steps' in train_params else 5
        gradient_accumulation_steps = train_params.pop('gradient_accumulation_steps') \
            if 'gradient_accumulation_steps' in train_params else 1
    
    else:
        # default
        epochs = 1
        batch_size = 64
        save_steps = 1000
        save_total_limit = 2
        logging_dir=None
        logging_steps = 5
        gradient_accumulation_steps = 1
    
    work_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # evaluation_strategy = "steps",
        # eval_steps=200,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,

        logging_steps=logging_steps,
        logging_dir=logging_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # learning_rate=5e-5,
        # disable_tqdm=True,
        # no_cuda=True,
        **train_params,
    )

    data_collator = DataCollatorWithPadding(tokenizer.bert_tokenizer)
    trainer = Trainer(
        model=model,
        args=work_args,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
