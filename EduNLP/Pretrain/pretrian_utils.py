from typing import Tuple
import traceback
import torch
import os
import json
from transformers import PretrainedConfig
from ..Tokenizer import get_tokenizer
from ..ModelZoo.utils import pad_sequence
from ..SIF import EDU_SPYMBOLS


class Vocab(object):
    def __init__(self, vocab_path=None, corpus_items=None, bos_token="[BOS]",
                 eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]",
                 specials=None, lower=False, trim_min_count=1, **argv):
        super(Vocab, self).__init__()

        self._tokens = []
        self.idx_to_token = dict()
        self.token_to_idx = dict()
        self.frequencies = dict()
        self._special_tokens = []

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        # 定义特殊词
        self._special_tokens = [self.bos_token, self.eos_token, self.pad_token, self.unk_token]
        if specials is not None:
            self._special_tokens += specials
        for st in self._special_tokens:
            self.add(st)
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
        return self.token_to_idx.get(token, self.unk_idx)

    def to_token(self, idx):
        return self.idx_to_token.get(idx, self.unk_token)

    def convert_sequence_to_idx(self, tokens, bos=False, eos=False):
        res = [self.to_idx(t) for t in tokens]
        if bos is True:
            res = [self.to_idx(self.bos_idx)] + res
        if eos is True:
            res = res + [self.to_idx(self.eos_idx)]
        return res

    def convert_sequence_to_token(self, idxs):
        return [self.to_token(i) for i in idxs]

    def set_vocab(self, corpus_items, lower=False, trim_min_count=1):
        word2cnt = dict()
        for item in corpus_items:
            for word in item:
                word = word.lower() if lower else word
                word2cnt[word] = word2cnt.get(word, 0) + 1
        words = [w for w, c in word2cnt.items() if c >= trim_min_count and w not in self._special_tokens]
        for token in words:
            self.add(token)

    def load_vocab(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as file:
            self._tokens = file.read().strip().split('\n')
            self.token_to_idx = {token: idx for idx, token in enumerate(self._tokens)}
            self.idx_to_token = {idx: token for idx, token in enumerate(self._tokens)}

    def save_vocab(self, vocab_path):
        with open(vocab_path, 'w', encoding='utf-8') as file:
            for i in range(self.vocab_size):
                token = self._tokens[i]
                file.write(f"{token}\n")

    def add(self, token):
        if token not in self._tokens:
            idx = len(self._tokens)
            self._tokens.append(token)
            self.idx_to_token[idx] = token
            self.token_to_idx[token] = idx

    def add_specials(self, tokens):
        for token in tokens:
            if token not in self._special_tokens:
                self._special_tokens += [token]
                self.add(token)


# to do: how to handle tokenizer with formulas or pictures.
class PretrainedTokenizer(object):
    def __init__(self, vocab_path=None, max_length=250, tokenize_method="char", add_specials: list = None, **argv):
        """
        Parameters
        ----------
        vocab_path: str
            default is None
        max_length: int
            default is 250, used to clip the sentence out of length
        tokenize_method: str
            default: "space"
            when text is already seperated by space, use "space"
            when text is raw string format, use Tokenizer defined in get_tokenizer(), such as "pure_text" and "text"
        """
        self._set_base_tokenizer(tokenize_method)

        specials = EDU_SPYMBOLS + [add_specials if add_specials is not None else None]
        self.max_length = max_length
        self.vocab = Vocab(vocab_path=vocab_path, specials=specials, **argv)

        config = {k: v for k, v in locals().items() if k not in ["self", "__class__", "vocab_path"]}
        self.config = PretrainedConfig.from_dict(config)

    def __call__(self, items: (list, str, dict), key=lambda x: x, padding=True,
                 return_tensors=True, return_text=False, **kwargs):
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
        """
        token_items = self.tokenize(items, key)
        if isinstance(items, str) or isinstance(items, dict):
            token_items = [token_items]

        seqs = [self.vocab.convert_sequence_to_idx(token_item,
                                                   bos=kwargs.get("bos", False),
                                                   eos=kwargs.get("eos", False)) for token_item in token_items]
        lengths = [len(seq) for seq in seqs]
        ret = {
            "seq_idx": pad_sequence(seqs, pad_val=self.vocab.pad_idx) if padding else seqs,
            "seq_len": lengths
        }

        if isinstance(items, str) or isinstance(items, dict):
            ret = {k: v[0] for k, v in ret.items()}
            token_items = token_items[0]

        if return_tensors:
            ret = {key: torch.as_tensor(val) for key, val in ret.items()}

        if return_text:
            ret["seq_token"] = token_items

        return ret

    def __len__(self):
        return len(self.vocab)

    def _set_base_tokenizer(self, tokenize_method):
        self.tokenize_method = tokenize_method
        if tokenize_method == "char":
            self.text_tokenizer = self._char_tokenizer
        elif tokenize_method == "space":
            self.text_tokenizer = self._space_tokenizer
        else:
            self.text_tokenizer = get_tokenizer(tokenize_method)

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

    def encode(self, items: Tuple[list, str, dict], key=lambda x: x):
        if isinstance(items, str) or isinstance(items, dict):
            return self.vocab.convert_sequence_to_idx(key(items))
        else:
            return[self.vocab.convert_sequence_to_idx(key(item)) for item in items]

    def decode(self, items: Tuple[list, str, dict], key=lambda x: x):
        if isinstance(items, str) or isinstance(items, dict):
            return self.vocab.convert_sequence_to_token(key(items))
        else:
            return[self.vocab.convert_sequence_to_token(key(item)) for item in items]

    def _tokenize(self, item: Tuple[str, dict], key=lambda x: x):
        try:
            token_item = next(self.text_tokenizer([item], key=key))
            if len(token_item) == 0:
                token_item = [self.vocab.unk_token]
            if len(token_item) > self.max_length:
                token_item = token_item[:self.max_length]
        except Exception:
            print("[debug]", item)
            msg = traceback.format_exc()
            print(msg)
        return token_item

    def _char_tokenizer(self, items, key=lambda x: x, **argv):
        for item in items:
            tokens = key(item).strip().split('')
            yield tokens

    def _space_tokenizer(self, items, key=lambda x: x, **argv):
        for item in items:
            tokens = key(item).strip().split(' ')
            yield tokens

    def set_vocab(self, items: list, key=lambda x: x, trim_min_count=1):
        """
        Parameters
        -----------
        items: list
            can be the list of str, or list of dict
        key: function
            determine how to get the text of each item
        """
        token_items = self.tokenize(items, key)
        self.vocab.set_vocab(corpus_items=token_items, trim_min_count=trim_min_count)
        return token_items

    @classmethod
    def from_pretrained(cls, tokenizer_config_dir, **argv):
        """
        Parameters:
        -----------
        tokenizer_config_dir: str
            must contain tokenizer_config.json and vocab.list
        """
        tokenizer_config_path = os.path.join(tokenizer_config_dir, "tokenizer_config.json")
        pretrained_vocab_path = os.path.join(tokenizer_config_dir, "vocab.txt")

        with open(tokenizer_config_path, "r", encoding="utf-8") as rf:
            tokenizer_config = json.load(rf)
            return cls(
                vocab_path=pretrained_vocab_path,
                **tokenizer_config)

    def save_pretrained(self, tokenizer_config_dir):
        """
        Parameters:
        -----------
        tokenizer_config_dir: str
            save tokenizer params in tokenizer_config.json and save words in vocab.list
        """
        tokenizer_config_path = os.path.join(tokenizer_config_dir, "tokenizer_config.json")
        vocab_path = os.path.join(tokenizer_config_dir, "vocab.txt")
        self.vocab.save_vocab(vocab_path)

        with open(tokenizer_config_path, "w", encoding="utf-8") as wf:
            json.dump(self.config.to_dict(), wf, ensure_ascii=False, indent=2)

    @property
    def vocab_size(self):
        return len(self.vocab)
