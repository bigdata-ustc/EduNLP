from typing import Optional, Union, List, Dict, Any, Iterable, Tuple
import traceback
import torch
import os
import json
import pandas as pd
from datasets import Dataset as HFDataset, load_from_disk
from torch.utils.data import Dataset
from ..Tokenizer import get_tokenizer
from ..ModelZoo.utils import pad_sequence
from ..SIF import EDU_SPYMBOLS


__all__ = ["EduVocab", "EduDataset", "PretrainedEduTokenizer"]


class EduVocab(object):
    """The vocabulary container for a corpus.

    Parameters
    ----------
    vocab_path : str, optional
        vocabulary path to initialize this container, by default None
    corpus_items : List[str], optional
        corpus items to update this vocabulary, by default None
    bos_token : str, optional
        token representing for the start of a sentence, by default "[BOS]"
    eos_token : str, optional
        token representing for the end of a sentence, by default "[EOS]"
    pad_token : str, optional
        token representing for padding, by default "[PAD]"
    unk_token : str, optional
        token representing for unknown word, by default "[UNK]"
    specials : List[str], optional
        spacials tokens in vocabulary, by default None
    lower : bool, optional
        wheather to lower the corpus items, by default False
    trim_min_count : int, optional
        the lower bound number for adding a word into vocabulary, by default 1
    """
    def __init__(self, vocab_path: str = None, corpus_items: List[str] = None, bos_token: str = "[BOS]",
                 eos_token: str = "[EOS]", pad_token: str = "[PAD]", unk_token: str = "[UNK]",
                 specials: List[str] = None, lower: bool = False, trim_min_count: int = 1, **argv):
        super(EduVocab, self).__init__()

        self._tokens = []
        self.idx_to_token = dict()
        self.token_to_idx = dict()
        self.frequencies = dict()
        # 定义特殊词
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self._special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

        if specials:
            self._special_tokens += specials
        for st in self._special_tokens:
            self._add(st)
        # 加载词典
        if vocab_path is not None:
            self.load_vocab(vocab_path)
        elif corpus_items is not None:
            self.set_vocab(corpus_items, lower, trim_min_count)

        self.bos_idx = self.token_to_idx[self.bos_token]
        self.eos_idx = self.token_to_idx[self.eos_token]
        self.pad_idx = self.token_to_idx[self.pad_token]
        self.unk_idx = self.token_to_idx[self.unk_token]

    def __len__(self):
        return len(self._tokens)

    @property
    def vocab_size(self):
        return len(self._tokens)

    @property
    def special_tokens(self):
        return self._special_tokens

    @property
    def tokens(self):
        return self._tokens

    def to_idx(self, token):
        """convert token to index"""
        return self.token_to_idx.get(token, self.unk_idx)

    def to_token(self, idx):
        """convert index to index"""
        return self.idx_to_token.get(idx, self.unk_token)

    def convert_sequence_to_idx(self, tokens, bos=False, eos=False):
        """convert sentence of tokens to sentence of indexs"""
        res = [self.to_idx(t) for t in tokens]
        if bos is True:
            res = [self.bos_idx] + res
        if eos is True:
            res = res + [self.eos_idx]
        return res

    def convert_sequence_to_token(self, idxs, **argv):
        """convert sentence of indexs to sentence of tokens"""
        return [self.to_token(i) for i in idxs]

    def set_vocab(self, corpus_items: List[str], lower: bool = False, trim_min_count: int = 1, silent=True):
        """Update the vocabulary with the tokens in corpus items

        Parameters
        ----------
        corpus_items : List[str], optional
            corpus items to update this vocabulary, by default None
        lower : bool, optional
            wheather to lower the corpus items, by default False
        trim_min_count : int, optional
            the lower bound number for adding a word into vocabulary, by default 1
        """
        word2cnt = dict()
        for item in corpus_items:
            for word in item:
                word = word.lower() if lower else word
                word2cnt[word] = word2cnt.get(word, 0) + 1
        words = [w for w, c in word2cnt.items() if c >= trim_min_count and w not in self._special_tokens]
        for token in words:
            self._add(token)
        if not silent:
            keep_word_cnts = sum(word2cnt[w] for w in words)
            all_word_cnts = sum(word2cnt.values())
            print(f"save words(trim_min_count={trim_min_count}): {len(words)}/{len(word2cnt)} = {len(words) / len(word2cnt):.4f}\
                  with frequency {keep_word_cnts}/{all_word_cnts}={keep_word_cnts / all_word_cnts:.4f}")

    def load_vocab(self, vocab_path: str):
        """Load the vocabulary from vocab_file

        Parameters
        ----------
        vocab_path : str
            path to save vocabulary file
        """
        with open(vocab_path, "r", encoding="utf-8") as file:
            self._tokens = file.read().strip().split('\n')
            self.token_to_idx = {token: idx for idx, token in enumerate(self._tokens)}
            self.idx_to_token = {idx: token for idx, token in enumerate(self._tokens)}

    def save_vocab(self, vocab_path: str):
        """Save the vocabulary into vocab_file

        Parameters
        ----------
        vocab_path : str
            path to save vocabulary file
        """
        with open(vocab_path, 'w', encoding='utf-8') as file:
            for i in range(self.vocab_size):
                token = self._tokens[i]
                file.write(f"{token}\n")

    def _add(self, token: str):
        if token not in self._tokens:
            idx = len(self._tokens)
            self._tokens.append(token)
            self.idx_to_token[idx] = token
            self.token_to_idx[token] = idx

    def add_specials(self, tokens: List[str]):
        """Add special tokens into vocabulary"""
        for token in tokens:
            if token not in self._special_tokens:
                self._special_tokens += [token]
                self._add(token)

    def add_tokens(self, tokens: List[str]):
        """Add tokens into vocabulary"""
        for token in tokens:
            self._add(token)


# to do: how to handle tokenizer with formulas or pictures.
class PretrainedEduTokenizer(object):
    """This base class is in charge of preparing the inputs for a model

    Parameters
    ----------
    vocab_path : str, optional
        _description_, by default None
    max_length : int, optional
        used to clip the sentence out of max_length, by default None
    tokenize_method : str, optional
        default: "space"
        - when text is already seperated by space, use "space"
        - when text is raw string format, use Tokenizer defined in get_tokenizer(), such as "pure_text" and "text"
    add_specials : Tuple[list, bool], optional
        by default None
        - For bool, it means whether to add EDU_SPYMBOLS to vocabulary
        - For list, it means the added special tokens besides EDU_SPYMBOLS
    """
    def __init__(self, vocab_path: str = None, max_length: int = 250, tokenize_method: str = "pure_text",
                 add_specials: Tuple[list, bool] = False, **argv):
        self._set_basic_tokenizer(tokenize_method, **argv)
        if isinstance(add_specials, bool):
            add_specials = EDU_SPYMBOLS if add_specials else []
        else:
            add_specials = EDU_SPYMBOLS + add_specials
        self.max_length = max_length
        self.vocab = EduVocab(vocab_path=vocab_path, specials=add_specials, **argv)

        self.config = {k: v for k, v in locals().items() if k not in ["self", "__class__", "vocab_path"]}

    def __call__(self, items: Tuple[list, str, dict], key=lambda x: x, padding: Tuple[bool, str] = True,
                 max_length=None, return_tensors=True, return_text=False, **kwargs) -> Dict[str, Any]:
        """
        Parameters
        ----------
        items: list or str or dict
            the question items
        key: function
            determine how to get the text of each item
        padding: bool
            whether to pad the seq_idx
        return_tensors: bool
            whether to return data as tensors (would ignore text tokens)
        return_text: bool
            whether to return text tokens

        Returns
        -------
        ret: dict
            {"seq_idx": None, "seq_len": None}
            or {"seq_token": None, seq_idx": None, "seq_len": None}.
            The shape of element is (batch, seq) or (batch,).

        Notes:
        -------
        Be Make sure Tokenizer output batched tensors by default
        """
        batch_max_length = None
        max_length = self.max_length if max_length is None else max_length
        if isinstance(padding, str):
            if padding == "max_length":
                batch_max_length = max_length
                padding = True
            elif padding == "longest":
                padding = True
            elif padding == "do_not_pad":
                padding = False
            else:
                raise ValueError("'padding' must be `bool` or `string` in ['max_length', 'longest', 'do_not_pad']")

        token_items = self.tokenize(items, key)
        if isinstance(items, dict) or isinstance(items, str):
            token_items = [token_items]
        if max_length is not None:
            token_items = [seq[:max_length] for seq in token_items]
        seqs = [self.vocab.convert_sequence_to_idx(token_item,
                                                   bos=kwargs.get("bos", False),
                                                   eos=kwargs.get("eos", False)) for token_item in token_items]
        lengths = [len(seq) for seq in seqs]
        ret = {
            "seq_idx": pad_sequence(seqs, pad_val=self.vocab.pad_idx, max_length=batch_max_length) if padding else seqs,
            "seq_len": lengths
        }
        if isinstance(items, dict) or isinstance(items, str):
            ret = {k: v[0] for k, v in ret.items()}
            token_items = token_items[0]
        if return_tensors:
            ret = {key: torch.as_tensor(val) for key, val in ret.items()}
        if return_text:
            ret["seq_token"] = token_items
        return ret

    def __len__(self):
        return len(self.vocab)

    def _set_basic_tokenizer(self, tokenize_method: str, **argv):
        self.tokenize_method = tokenize_method
        self.text_tokenizer = get_tokenizer(tokenize_method, **argv)

    def tokenize(self, items: Tuple[list, str, dict], key=lambda x: x, **kwargs):
        """
        Parameters
        ----------
        items: list or str or dict
            the question items
        key: function
            determine how to get the text of each item

        Returns
        -------
        tokens: list
            the token of items
        """
        if isinstance(items, str) or isinstance(items, dict):
            return self._tokenize(items, key=key)
        else:
            return [self._tokenize(item, key=key) for item in items]

    def encode(self, items: Tuple[str, dict, List[str], List[dict]], key=lambda x: x, **argv):
        if isinstance(items, str) or isinstance(items, dict):
            return self.vocab.convert_sequence_to_idx(key(items), **argv)
        else:
            return [self.vocab.convert_sequence_to_idx(key(item), **argv) for item in items]

    def decode(self, token_ids: list, key=lambda x: x, **argv):
        if isinstance(token_ids[0], list):
            return [self.vocab.convert_sequence_to_token(key(item), **argv) for item in token_ids]
        else:
            return self.vocab.convert_sequence_to_token(key(token_ids), **argv)

    def _pad(self):
        raise NotImplementedError

    def _tokenize(self, item: Tuple[str, dict], key=lambda x: x):
        token_item = self.text_tokenizer._tokenize(item, key=key)
        if len(token_item) == 0:
            token_item = [self.vocab.unk_token]
        if len(token_item) > self.max_length:
            token_item = token_item[:self.max_length]
        return token_item

    @classmethod
    def from_pretrained(cls, tokenizer_config_dir: str, **argv):
        """Load tokenizer from local files

        Parameters:
        -----------
        tokenizer_config_dir: str
            The dir path containing tokenizer_config.json and vocab.list
        """
        tokenizer_config_path = os.path.join(tokenizer_config_dir, "tokenizer_config.json")
        pretrained_vocab_path = os.path.join(tokenizer_config_dir, "vocab.txt")

        with open(tokenizer_config_path, "r", encoding="utf-8") as rf:
            tokenizer_config = json.load(rf)
            tokenizer_config.update(argv)
            return cls(
                vocab_path=pretrained_vocab_path,
                **tokenizer_config)

    def save_pretrained(self, tokenizer_config_dir: str):
        """Save tokenizer into local files

        Parameters:
        -----------
        tokenizer_config_dir: str
            save tokenizer params in `/tokenizer_config.json` and save words in `/vocab.list`
        """
        if not os.path.exists(tokenizer_config_dir):
            os.makedirs(tokenizer_config_dir, exist_ok=True)
        tokenizer_config_path = os.path.join(tokenizer_config_dir, "tokenizer_config.json")
        vocab_path = os.path.join(tokenizer_config_dir, "vocab.txt")
        self.vocab.save_vocab(vocab_path)

        with open(tokenizer_config_path, "w", encoding="utf-8") as wf:
            json.dump(self.config, wf, ensure_ascii=False, indent=2)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def set_vocab(self, items: list, key=lambda x: x, lower: bool = False,
                  trim_min_count: int = 1, do_tokenize: bool = True):
        """Update the vocabulary with the tokens in corpus items

        Parameters
        ----------
        items: list
            can be the list of str, or list of dict
        key: function, optional
            determine how to get the text of each item
        lower : bool, optional
            wheather to lower the corpus items, by default False
        trim_min_count : int, optional
            the lower bound number for adding a word into vocabulary, by default 1
        do_tokenize : bool, optional
            wheather tokenize items before updating vocab, by default True

        Returns
        -------
        list
            token_items
        """
        token_items = self.tokenize(items, key) if do_tokenize else [key(item) for item in items]
        self.vocab.set_vocab(corpus_items=token_items, trim_min_count=trim_min_count, lower=lower)
        return token_items

    def add_specials(self, tokens):
        """Add special tokens into vocabulary"""
        self.vocab.add_specials(tokens)

    def add_tokens(self, tokens):
        """Add tokens into vocabulary"""
        self.vocab.add_tokens(tokens)


class EduDataset(Dataset):
    """The base class implements a Dataset, which package the `datasets.Dataset`
    and provide more convenience, including parallel preprocessing, offline loadding and so on.

    Parameters
    ----------
    tokenizer :
        PretrainedEduTokenizer or model-specific Pretrained Tokenizer
    ds_disk_path : HFDataset, optional
        the dataset_path to save dataset used by `datasets.Dataset`, by default None
    items : Union[List[dict], List[str]], optional
        input items to process, by default None
    stem_key : str, optional
        the content of items to process, by default "text"
    label_key : Optional[str], optional
        the labels of items to process, by default None
    feature_keys : Optional[List[str]], optional
        the additional features of items to remain, by default None
    num_processor : int, optional
        specific the number of cpus for parallel speedup, by default None
    """
    def __init__(self, tokenizer, ds_disk_path: HFDataset = None,
                 items: Union[List[dict], List[str]] = None,
                 stem_key: str = "text", label_key: Optional[str] = None,
                 feature_keys: Optional[List[str]] = None,
                 num_processor: int = None, **argv):
        self.tokenizer = tokenizer
        feature_keys = [] if feature_keys is None else feature_keys
        if items is not None:
            assert ds_disk_path is None
            if isinstance(items[0], dict):
                assert stem_key is not None
                raw_columns = set(items[0].keys())
            if isinstance(items[0], str):
                assert stem_key is None and label_key is None
                stem_key = "text"
                raw_columns = [stem_key]
            work_columns = set([stem_key] + feature_keys + ([label_key] if label_key is not None else []))
            redundant_columns = raw_columns - work_columns
            # 在线预处理特征
            items = items if isinstance(items[0], dict) else [{"text": i} for i in items]
            df = pd.DataFrame(items)
            df.drop(columns=list(redundant_columns), inplace=True)
            self.ds = HFDataset.from_pandas(df)
            """Note: map will break down for super large data which is greater than 4GB """
            self.ds = self.ds.map(lambda sample: tokenizer(sample[stem_key], return_tensors=False),
                                  num_proc=num_processor,
                                  batched=True, batch_size=1000)
            remove_columns = [stem_key]
        else:
            # 离线加载工作特征
            assert ds_disk_path is not None
            self.ds = load_from_disk(ds_disk_path)
            reserve_columns = list(tokenizer("edunlp", return_tensors=False).keys())\
                + feature_keys + ([label_key] if label_key is not None else [])
            remove_columns = list(set(self.ds.column_names) - set(reserve_columns))

        # 工作特征
        self.work_ds = self.ds.remove_columns(remove_columns) if len(remove_columns) > 0 else self.ds
        if label_key is not None:
            self.work_ds = self.work_ds.rename_columns({
                label_key: "labels",
            })

    def __getitem__(self, index):
        return self.work_ds[index]

    def __len__(self):
        return self.work_ds.num_rows

    def to_disk(self, ds_disk_path):
        """Save the processed dataset into local files"""
        self.ds.save_to_disk(ds_disk_path)

    def collect_fn(self):
        raise NotImplementedError
