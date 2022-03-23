from msilib.schema import Error
import os
from re import A
from sre_constants import ASSERT
from cv2 import erode
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..ModelZoo.DisenQNet.DisenQNet import DisenQNet, ConceptModel
from ..Tokenizer import get_tokenizer
from ..ModelZoo.utils import load_items

from ..SIF import Symbol, FORMULA_SYMBOL, FIGURE_SYMBOL, QUES_MARK_SYMBOL, TAG_SYMBOL, SEP_SYMBOL
import json
import logging
import warnings
from gensim.models import Word2Vec


def check_num(s):
    # (1/2) -> 1/2
    if s.startswith('(') and s.endswith(')'):
        if s == "()":
            return False
        else:
            s = s[1:-1]
    # -1/2 -> 1/2
    if s.startswith('-') or s.startswith('+'):
        if s == '-' or s == '+':
            return False
        else:
            s = s[1:]
    # 1/2
    if '/' in s:
        if s == '/':
            return False
        else:
            sep_index = s.index('/')
            is_num = s[:sep_index].isdigit() and s[sep_index + 1:].isdigit()
            return is_num
    else:
        # 1.2% -> 1.2
        if s.endswith('%'):
            if s == '%':
                return False
            else:
                s = s[:-1]
        # 1.2
        if '.' in s:
            if s == '.':
                return False
            else:
                try:
                    float(s)
                    return True
                except Exception:
                    return False
        # 12
        else:
            is_num = s.isdigit()
            return is_num


def list_to_onehot(item_list, item2index):
    onehot = np.zeros(len(item2index)).astype(np.int64)
    for c in item_list:
        onehot[item2index[c]] = 1
    return onehot


def load_list(path):
    with open(path, "rt", encoding="utf-8") as file:
        items = file.read().strip().split('\n')
    item2index = {item: index for index, item in enumerate(items)}
    return item2index


def save_list(item2index, path):
    item2index = sorted(item2index.items(), key=lambda kv: kv[1])
    items = [item for item, _ in item2index]
    with open(path, "wt", encoding="utf-8") as file:
        file.write('\n'.join(items))
    return


class DisenQTokenizer(object):
    """
    Examples
    --------
    >>> tokenizer = DisenQTokenizer()
    >>> test_items = [{
    ...     "question_id": "946",
    ...     "content": "甲 数 除以 乙 数 的 商 是 1.5 ， 如果 甲 数 增加 20 ， 则 甲 数 是 乙 的 4 倍 ． 原来 甲 数 = ．",
    ...     "knowledge": ["*", "-", "/"], "difficulty": 0.2, "length": 7}]
    >>> tokenizer.set_vocab(test_items,
    ...     trim_min_count=1, key=lambda x: x["content"], silent=True)
    >>> token_items = [tokenizer(i, key=lambda x: x["content"]) for i in test_items]
    >>> print(token_items[0].keys())
    dict_keys(['content_idx', 'content_len'])
    """
    def __init__(self, vocab_path=None, max_length=250, tokenize_method="space",
                 num_token="<num>", unk_token="<unk>", pad_token="<pad>", *args, **argv):
        """
        Parameters
        ----------
        vocab_path: str
            default is None
        max_length: int
            default is 250
        tokenize_method: str
            default: "space"
            when text is already seperated by space, use "space"
            when text is row text format, use Tokentizer defined in get_tokenizer(), such as "pure_text" and "text"
        """
        super(DisenQTokenizer, self).__init__(*args)
        self.tokenize_method = tokenize_method
        self.text_tokenizer = get_tokenizer(tokenize_method) if tokenize_method != "space" else self._space_toeknzier

        self.num_token = num_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.max_length = max_length
        if vocab_path is not None:
            self.load_vocab(vocab_path)
            self.secure = True
        else:
            self.secure = False

    def __call__(self, item: (str, dict), key=lambda x: x, padding=True,
                 return_tensors=True, return_text=False, *args, **kwargs):
        """
        item: str or dict
            the question item
        key: function
            determine how to get the text of each item
        padding: bool
            whether to pad the content_idx
        return_tensors: bool
            whether to return data as tensors (would ignore text tokens)
        return_text: bool
            whether to return text tokens
        """
        token_item = self.tokenize(item, key)
        indexs = [self.word2index.get(w, self.word2index[self.unk_token]) for w in item]
        length = len(indexs)
        ret = {
            "content_idx": self.padding(indexs, self.max_length) if padding else indexs,
            "content_len": length
        }

        if return_tensors:
            return {key: torch.as_tensor(val) for key, val in ret.items()}
        elif return_text:
            ret["content"] = token_item

        return ret

    def tokenize(self, item: (str, dict), key=lambda x: x, *args, **kwargs):
        if isinstance(item, str) or isinstance(item, dict):
            return self._tokenize(item, key)
        else:
            raise ValueError("items should be str or list!")

    def padding(self, idx, max_length):
        padding_idx = idx + [self.word2index[self.pad_token]] * (max_length - len(idx))
        return padding_idx

    def _space_toeknzier(self, items, key=lambda x: x):
        for item in items:
            yield key(item).strip().split(' ')

    def _tokenize(self, item, key=lambda x: x):
        if not self.secure:
            raise Exception("Must set the vocab first before tokenize item (either set_vocab() or load_vocab() )")
        item = next(self.text_tokenizer([item], key=key))
        stop_words = set("\n\r\t .,;?\"\'。．，、；？“”‘’（）")
        token_text = [w for w in item if w != '' and w not in stop_words]
        if len(token_text) == 0:
            token_text = [self.unk_token]
        if len(token_text) > self.max_length:
            token_text = token_text[:self.max_length]
        token_text = [self.num_token if check_num(w) else w for w in token_text]
        return token_text

    def load_vocab(self, path):
        with open(path, "rt", encoding="utf-8") as file:
            self.words = file.read().strip().split('\n')
            self.word2index = {word: index for index, word in enumerate(self.words)}

    def set_vocab(self, items: list, key=lambda x: x, trim_min_count=50, silent=True):
        """
        Parameters
        -----------
        items: list
            can be the list of str, or list of dict
        key: function
            determine how to get the text of each item
        """
        self.secure = True
        word2cnt = dict()
        token_items = list()
        for item in items:
            token_item = self.tokenize(item, key=key)
            token_items.append(token_item)
        for item in token_items:
            for w in item:
                word2cnt[w] = word2cnt.get(w, 0) + 1
        ctrl_tokens = [self.num_token, self.unk_token, self.pad_token]
        words = [w for w, c in word2cnt.items() if c >= trim_min_count and w not in ctrl_tokens]
        if not silent:
            keep_word_cnts = sum(word2cnt[w] for w in words)
            all_word_cnts = sum(word2cnt.values())
            print(f"save words({trim_min_count}): {len(words)}/{len(word2cnt)} = {len(words)/len(word2cnt):.4f}\
                  with frequency {keep_word_cnts}/{all_word_cnts}={keep_word_cnts/all_word_cnts:.4f}")

        self.words = ctrl_tokens + sorted(words)
        self.word2index = {word: index for index, word in enumerate(self.words)}

    def save_vocab(self, save_vocab_path):
        save_list(self.word2index, save_vocab_path)  # only save words

    @classmethod
    def from_pretrained(cls, tokenzier_config_dir):
        """
        Parameters:
        -----------
        tokenzier_config_dir: str
            must contain tokenzier_config.json and vocab.list
        """
        with open(tokenzier_config_dir, "r", encoding="utf-8") as rf:
            tokenzier_config = json.load(rf)
            return cls(
                vocab_path=tokenzier_config["vocab_path"], max_length=tokenzier_config["max_length"],
                tokenize_method=tokenzier_config["tokenize_method"], num_token=tokenzier_config["num_token"],
                unk_token=tokenzier_config["unk_token"], pad_token=tokenzier_config["pad_token"])

    def save_pretrained(self, tokenzier_config_dir):
        """
        Parameters:
        -----------
        tokenzier_config_dir: str
            save tokenzier params in tokenzier_config.json and save words in vocab.list
        """
        tokenzier_config_path = os.path.join(tokenzier_config_dir, "tokenzier_config.json")
        save_vocab_path = os.path.join(tokenzier_config_dir, "vocab.list")
        tokenzier_params = {
            "tokenize_method": self.tokenize_method,
            "num_token": self.num_token,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "max_length": self.max_length,
            "vocab_path": save_vocab_path,
        }
        self.save_vocab(save_vocab_path)
        with open(tokenzier_config_path, "w", encoding="utf-8") as wf:
            json.dump(tokenzier_params, wf, ensure_ascii=False, indent=2)

    @property
    def vocab_size(self):
        return len(self.word2index)


class QuestionDataset(Dataset):
    """
    Question dataset including text, length, concept Tensors
    """
    def __init__(self, items, predata_dir, max_length, dataset_type, silent=False, embed_dim=128, trim_min_count=50):
        """
        Parameters
        ----------
        items: list
            the raw question in json format
        predata_dir: str
            the dir to save or load the predata (include wv.th, vocab.list, concept.list )
        max_length: int
            the max length for text truncation
        """
        super(QuestionDataset, self).__init__()
        self.silent = silent
        self.dataset_type = dataset_type
        self.wv_path = os.path.join(predata_dir, "wv.th")
        self.word_list_path = os.path.join(predata_dir, "vocab.list")
        self.concept_list_path = os.path.join(predata_dir, "concept.list")

        file_num = sum(map(lambda x: os.path.exists(x), [self.wv_path, self.word_list_path, self.concept_list_path]))
        if file_num > 0 and file_num < 3:
            warnings.warn("Some files are missed in predata_dir(which including wv.th, vocab.list, and concept.list)")

        init = file_num != 3
        if not init:
            # load word, concept list
            self.word2index = load_list(self.word_list_path)
            self.concept2index = load_list(self.concept_list_path)
            self.word2vec = torch.load(self.wv_path)

            self.disen_tokenzier = DisenQTokenizer(vocab_path=self.word_list_path, max_length=max_length)
            if not silent:
                print(f"load vocab from {self.word_list_path}")
                print(f"load concept from {self.concept_list_path}")
                print(f"load word2vec from {self.wv_path}")
        else:
            self.disen_tokenzier = DisenQTokenizer(max_length=max_length)
            self.disen_tokenzier.set_vocab(items, key=lambda x: x["content"],
                                           trim_min_count=trim_min_count)
            self.disen_tokenzier.save_pretrained(predata_dir)
            if not silent:
                print(f"save vocab to {self.word_list_path}")
            self.word2index = self.disen_tokenzier.word2index

        self.num_token = self.disen_tokenzier.num_token  # "<num>"
        self.unk_token = self.disen_tokenzier.unk_token  # "<unk>"
        self.pad_token = self.disen_tokenzier.pad_token  # "<pad>"
        self.unk_idx = self.word2index[self.unk_token]

        # load dataset, init construct word and concept list
        if dataset_type == "train":
            self.dataset = self.process_dataset(items, trim_min_count, embed_dim, init)
        elif dataset_type == "test":
            self.dataset = self.process_dataset(items, trim_min_count, embed_dim, False)

        if not silent:
            print("processing raw data for QuestionDataset...")

        self.vocab_size = len(self.word2index)
        self.concept_size = len(self.concept2index)
        if dataset_type == "train" and not silent:
            print(f"vocab size: {self.vocab_size}")
            print(f"concept size: {self.concept_size}")
        return

    def process_dataset(self, items, trim_min_count, embed_dim, init=False, text_key=lambda x: x["content"]):
        # make items in standard format
        for i, item in enumerate(items):
            token_data = self.disen_tokenzier(item, key=text_key, return_tensors=False, return_text=True, padding=False)
            items[i]["content_idx"] = token_data["content_idx"]
            items[i]["content_len"] = token_data["content_len"]
            items[i]["content"] = token_data["content"]

        if init:
            # construct concept list
            if not os.path.exists(self.concept_list_path):
                concepts = set()
                for data in items:
                    concept = data["knowledge"]
                    for c in concept:
                        if c not in concepts:
                            concepts.add(c)
                concepts = sorted(concepts)
                self.concept2index = {concept: index for index, concept in enumerate(concepts)}
                save_list(self.concept2index, self.concept_list_path)
                if not self.silent:
                    print(f"save concept to {self.concept_list_path}")
            else:
                self.concept2index = load_list(self.concept_list_path)
            # word2vec
            if not os.path.exists(self.wv_path):
                words = self.disen_tokenzier.words
                corpus = list()
                word_set = set(words)
                for data in items:
                    text = [w if w in word_set else self.unk_token for w in data["content"]]
                    corpus.append(text)
                wv = Word2Vec(corpus, vector_size=embed_dim, min_count=trim_min_count).wv
                # 按照 vocab 中的词序 来保存
                # wv_list = [wv[w] if w in wv.key_to_index else np.random.rand(embed_dim) for w in words]
                ctrl_tokens = [self.num_token, self.unk_token, self.pad_token]
                wv_list = [wv[w] if w not in ctrl_tokens else np.random.rand(embed_dim) for w in words]
                self.word2vec = torch.tensor(wv_list)
                torch.save(self.word2vec, self.wv_path)
                if not self.silent:
                    print(f"save word2vec to {self.wv_path}")
            else:
                self.word2vec = torch.load(self.wv_path)
        return items

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # map word to index
        data = self.dataset[index]
        text = data["content_idx"]
        length = data["content_len"]
        concept = list_to_onehot(data["knowledge"], self.concept2index)
        return text, length, concept

    def collate_data(self, batch_data):
        pad_idx = self.word2index[self.pad_token]
        text, length, concept = list(zip(*batch_data))
        max_length = max(length)
        text = [t + [pad_idx] * (max_length - len(t)) for t in text]
        text = torch.tensor(text)
        length = torch.tensor(length)
        concept = torch.tensor(concept)
        return text, length, concept


def train_disenQNet(train_items, output_dir, predata_dir, train_params=None, test_items=None, silent=False):
    """
    Parameters
    ----------
    train_items: list
        the raw train question list
    output_dir: str
        the path to save the model
    predata_dir: str
        the dirname of predata_dir(include )
    train_params: dict
        the training parameters
    test_items: list or None
        the raw test question list, default is None

    Examples
    ----------
    >>> train_data = load_items("tests/test_vec/disenq_train.json")[:100]
    >>> test_data = load_items("tests/test_vec/disenq_test.json")[:100]
    >>> train_disenQNet(train_data,
    ... "examples/test_model/data/disenq","examples/test_model/data/disenq", silent=True) #doctest: +ELLIPSIS
    """
    # dataset
    default_train_params = {
        "trim_min": 2,
        "max_len": 250,

        "hidden": 128,
        "dropout": 0.2,
        "pos_weight": 1,

        "cp": 1.5,
        "mi": 1.0,
        "dis": 2.0,

        "epoch": 1,
        "batch": 64,
        "lr": 1e-3,
        "step": 20,
        "gamma": 0.5,
        "warm_up": 1,
        "adv": 10,
        "device": "cpu",
    }
    if train_params is not None:
        default_train_params.update(train_params)
    train_params = default_train_params

    train_dataset = QuestionDataset(train_items, predata_dir, train_params["max_len"], "train", silent=silent,
                                    embed_dim=train_params["hidden"], trim_min_count=train_params["trim_min"])
    train_dataloader = DataLoader(train_dataset, batch_size=train_params["batch"],
                                  shuffle=True, collate_fn=train_dataset.collate_data)
    if test_items is not None:
        test_dataset = QuestionDataset(test_items, predata_dir, train_params["max_len"], "test", silent=silent,
                                       embed_dim=train_params["hidden"], trim_min_count=train_params["trim_min"])
        test_dataloader = DataLoader(test_dataset, batch_size=train_params["batch"],
                                     shuffle=False, collate_fn=test_dataset.collate_data)
    else:
        test_dataloader = None
    vocab_size = train_dataset.vocab_size
    concept_size = train_dataset.concept_size
    wv = train_dataset.word2vec

    # model
    disen_q_net = DisenQNet(vocab_size, concept_size, train_params["hidden"], train_params["dropout"],
                            train_params["pos_weight"], train_params["cp"], train_params["mi"], train_params["dis"], wv)

    disen_q_net.train(train_dataloader, test_dataloader, train_params["device"], train_params["epoch"],
                      train_params["lr"], train_params["step"], train_params["gamma"],
                      train_params["warm_up"], train_params["adv"], silent=silent)
    disen_q_net_path = os.path.join(output_dir, "disen_q_net.th")
    disen_q_net.save(disen_q_net_path)
