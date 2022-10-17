import os
import json
from transformers.file_utils import TensorType
from transformers import AutoTokenizer
from typing import List, Optional, Union, Tuple
from ..SIF import EDU_SPYMBOLS
from ..Tokenizer import get_tokenizer


class TokenizerForHuggingface(object):
    """
    Parameterss
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
    >>> tokenizer = TokenizerForHuggingface(add_special_tokens=True)
    >>> item = "有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"
    >>> token_item = tokenizer(item)
    >>> print(token_item.input_ids[:10])
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
    >>> tokenizer = TokenizerForHuggingface.from_pretrained('test_dir') # doctest: +SKIP
    """
    def __init__(self, pretrained_model="bert-base-chinese", max_length=512, tokenize_method: str = "pure_text",
                 add_specials: Tuple[List[str], bool] = False, **argv):
        self._set_basic_tokenizer(tokenize_method, **argv)
        if isinstance(add_specials, bool):
            add_specials = EDU_SPYMBOLS if add_specials is True else []
        else:
            add_specials = EDU_SPYMBOLS + add_specials
        self._special_tokens = set()
        self.max_length = max_length
        self.bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.add_specials(add_specials)
        config = {k: v for k, v in locals().items() if k not in ["self", "__class__", "pretrained_model", "argv"]}
        config.update(argv)
        self.config = config

    def __call__(self, items: Tuple[list, str, dict], key=lambda x: x, padding=True,
                 return_tensors: Optional[Tuple[str, TensorType, bool]] = True, **kwargs):
        if isinstance(items, list):
            text = [self._pre_tokenize(key(i)) for i in items]
        else:
            text = self._pre_tokenize(key(items))

        if isinstance(return_tensors, bool):
            return_tensors = "pt" if return_tensors is True else None
        encodes = self.bert_tokenizer(text, truncation=True, padding=padding, max_length=self.max_length,
                                      return_tensors=return_tensors)
        return encodes

    def __len__(self):
        return len(self.bert_tokenizer)

    def _set_basic_tokenizer(self, tokenize_method: str = None, **argv):
        self.tokenize_method = tokenize_method
        if self.tokenize_method is not None:
            self.text_tokenizer = get_tokenizer(tokenize_method, **argv)
        else:
            self.text_tokenizer = None

    def _pre_tokenize(self, text: Union[str, dict]):
        if self.text_tokenizer is not None:
            text = self.text_tokenizer._tokenize(text)
            text = " ".join(text)
        return text

    def tokenize(self, items: Union[list, str, dict], key=lambda x: x, **kwargs):
        if isinstance(items, list):
            texts = [self._tokenize(key(i)) for i in items]
            return texts
        else:
            return self._tokenize(key(items))

    def encode(self, items: Tuple[str, dict, List[str], List[dict]], key=lambda x: x, **argv):
        if isinstance(items, str) or isinstance(items, dict):
            return self.bert_tokenizer.encode(key(items), **argv)
        else:
            return [self.bert_tokenizer.encode(key(item), **argv) for item in items]

    def decode(self, token_ids: list, key=lambda x: x, **argv):
        if isinstance(token_ids[0], list):
            return [self.bert_tokenizer.decode(key(item), **argv) for item in token_ids]
        else:
            return self.bert_tokenizer.decode(key(token_ids), **argv)

    def _tokenize(self, item: Union[str, dict], key=lambda x: x, **kwargs):
        item = self._pre_tokenize(key(item))
        return self.bert_tokenizer.tokenize(item, **kwargs)

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

    def set_vocab(self, items: Tuple[List[str], List[dict]], key=lambda x: x, lower=False,
                  trim_min_count=1, do_tokenize=True):
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
            tokens = self._pre_tokenize(key(item)).split() if do_tokenize else key(item)
            if not tokens:
                continue
            for word in tokens:
                word = word.lower() if lower else word
                word2cnt[word] = word2cnt.get(word, 0) + 1
        added_words = [w for w, c in word2cnt.items() if c >= trim_min_count]
        added_num = self.add_tokens(added_words)
        return added_words, added_num

    def add_specials(self, added_spectials: List[str]):
        for tok in added_spectials:
            self._special_tokens.add(tok)

        return self.bert_tokenizer.add_special_tokens({'additional_special_tokens': added_spectials})

    def add_tokens(self, added_tokens: List[str]):
        return self.bert_tokenizer.add_tokens(added_tokens)
