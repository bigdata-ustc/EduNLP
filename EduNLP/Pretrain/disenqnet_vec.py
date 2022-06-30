import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ..ModelZoo.disenqnet.disenqnet import DisenQNet
from ..Tokenizer import get_tokenizer
from ..ModelZoo.utils import load_items, pad_sequence
import json
import warnings
from typing import Dict, List, Tuple
from .pretrian_utils import PretrainedEduTokenizer
from .gensim_vec import train_vector
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
            try:
                float(s)
                return True
            except ValueError:
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


def load_list_to_dict(path):
    with open(path, "rt", encoding="utf-8") as file:
        items = file.read().split('\n')
    item2index = {item: index for index, item in enumerate(items)}
    return item2index

def save_dict_to_list(item2index, path):
    item2index = sorted(item2index.items(), key=lambda kv: kv[1])
    items = [item for item, _ in item2index]
    with open(path, "wt", encoding="utf-8") as file:
        file.write('\n'.join(items))
    return

class DisenQTokenizer(PretrainedEduTokenizer):
    """
    Examples
    --------
    >>> tokenizer = DisenQTokenizer()
    >>> test_items = [{
    ...     "content": "甲 数 除以 乙 数 的 商 是 1.5 ， 如果 甲 数 增加 20 ， 则 甲 数 是 乙 的 4 倍 ． 原来 甲 数 = ．",
    ...     "knowledge": ["*", "-", "/"], "difficulty": 0.2, "length": 7}]
    >>> tokenizer.set_vocab(test_items,
    ...     trim_min_count=1, key=lambda x: x["content"], silent=True)
    >>> token_items = [tokenizer(i, key=lambda x: x["content"]) for i in test_items]
    >>> print(token_items[0].keys())
    dict_keys(['seq_idx', 'seq_len'])
    """
    def __init__(self, vocab_path=None, max_length=250, tokenize_method="space", 
                 num_token="[NUM]", **argv):
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
        num_token: str
        """
        add_specials = argv.get("add_specials", []) + [num_token]
        super().__init__(vocab_path, max_length, tokenize_method, add_specials=add_specials, **argv)
        self.num_token = num_token
        self.config.update({
            "num_token": num_token
        })

    def _tokenize(self, item: Tuple[str, dict], key=lambda x: x):
        token_item = next(self.text_tokenizer([item], key=key))
        if len(token_item) == 0:
            token_item = [self.unk_token]
        if len(token_item) > self.max_length:
            token_item = token_item[:self.max_length]

        token_item = [self.num_token if check_num(w) else w for w in token_item]
        return token_item

    def _space_tokenizer(self, items, key=lambda x: x):
        stop_words = set("\n\r\t .,;?\"\'。．，、；？“”‘’（）")
        for item in items:
            tokens = key(item).strip().split(' ')
            token_item = [w for w in tokens if w != '' and w not in stop_words]
            yield token_item


def preprocess_dataset(pretrained_dir, disen_tokenizer, items, data_formation, trim_min_count=None, embed_dim=None, w2v_params=None, silent=False):
        default_data_formation = {
            "content": "content",
            "knowledge": "knowledge"
        }
        if data_formation is not None:
            default_data_formation.update(data_formation)
        data_formation = default_data_formation

        default_w2v_params = {
            "workers": 1,
        }
        if w2v_params is not None:
            default_w2v_params.update(w2v_params)
        w2v_params = default_w2v_params

        concept_list_path = os.path.join(pretrained_dir, "concept.list")
        vocab_path = os.path.join(pretrained_dir, "vocab.list")
        wv_path = os.path.join(pretrained_dir, "wv.th")

        file_num = sum(map(lambda x: os.path.exists(x), [wv_path, concept_list_path]))
        if file_num > 0 and file_num < 3:
            warnings.warn("Some files are missed in pretrained_dir(which including wv.th, vocab.list, and concept.list)",
                          UserWarning)
        
        if not os.path.exists(vocab_path):
            # load word
            token_items = disen_tokenizer.set_vocab(items, key=lambda x: x[data_formation["content"]], trim_min_count=trim_min_count)
            disen_tokenizer.save_pretrained(pretrained_dir)
            if not silent:
                print(f"save vocab to {vocab_path}")
        else:
            if not silent:
                print(f"load vocab from {vocab_path}")

        # construct concept list
        if not os.path.exists(concept_list_path):
            concepts = set()
            for data in items:
                concept = data[data_formation["knowledge"]]
                for c in concept:
                    if c not in concepts:
                        concepts.add(c)
            concepts = sorted(concepts)
            concept_to_idx = {concept: index for index, concept in enumerate(concepts)}
            save_dict_to_list(concept_to_idx, concept_list_path)
            if not silent:
                print(f"save concept to {concept_list_path}")
        else:
            print(f"load concept from {concept_list_path}")
            concept_to_idx = load_list_to_dict(concept_list_path)

        # word2vec
        if not os.path.exists(wv_path):
            words = disen_tokenizer.vocab.tokens
            unk_token = disen_tokenizer.vocab.unk_token
            corpus = list()
            word_set = set(words)
            for text in token_items:
                text = [w if w in word_set else unk_token for w in text]
                corpus.append(text)
            wv = Word2Vec(corpus, vector_size=embed_dim, min_count=trim_min_count, **w2v_params).wv
            # 按照 vocab 中的词序 来保存
            wv_list = [wv[w] if w in wv.key_to_index else np.random.rand(embed_dim) for w in words]
            word2vec = torch.tensor(wv_list)
            torch.save(word2vec, wv_path)
            if not silent:
                print(f"save word2vec to {wv_path}")
        else:
            print(f"load word2vec from {wv_path}")
            word2vec = torch.load(wv_path)
        return disen_tokenizer, concept_to_idx, word2vec


class DisenQDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer: DisenQTokenizer, mode="train", concept_to_idx=None, **argv):
        """
        Parameters
        ----------
        texts: list
        tokenizer: DisenQTokenizer
        max_length: int, optional, default=128
        """
        super(DisenQDataset, self).__init__()
        self.tokenizer = tokenizer
        self.concept_to_idx = concept_to_idx
        self.mode = mode
        self.items = items
        self.max_length = tokenizer.max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        item = self.texts[index]
        ret = self.tokenizer(item, padding=False, key=lambda x: x,
                              return_tensors=True, return_text=False)
        if self.mode in ["train", "val"]:
            ret["know_idx"] = self.concept_to_idx[item["knowledge"]]

    def collate_fn(self, batch_data):
        pad_idx = self.tokenizer.vocab.pad_idx
        first =  batch_data[0]
        batch = {
            k: [item[k] for item in batch_data] for k in first.keys()
        }
        batch["seq_idx"] = pad_sequence(batch["seq_idx"], pad_val=pad_idx)

        batch = {key: torch.as_tensor(val) for key, val in batch[0].items()}
        return batch


def train_disenqnet(train_items, output_dir, pretrained_dir=None,
                    train_params=None, test_items=None, silent=False, data_formation=None):
    """
    Parameters
    ----------
    train_items: list
        the raw train question list
    disen_tokenizer: DisenQTokenizer
        the initial DisenQTokenizer use for training.
    output_dir: str
        the path to save the model
    pretrained_dir: str
        the dirname to load or save predata (including wv.th, vocab.list and concept.list)
    train_params: dict, defaults to None
        the training parameters for data, model and trianer.
        - "trim_min": int
            data param, the trim_min_count for vocab and word2vec, by default 2
        - "w2v_workers": int
            data param, the number of workers for word2vec, by default 1
        - "hidden": int
            model param, by default 128
        - "dropout": float
            model param, dropout rate, by default 0.2
        - "pos_weight": int
            model param, positive sample weight in unbalanced multi-label concept classifier, by default 1
        - "cp": float
            model param, weight of concept loss, by default 1.5
        - "mi": float
            model param, weight of mutual information loss, by default 1.0
        - "dis": float
            model param, weight of disentangling loss, by default 2.0
        - "epoch": int
            train param, number of epoch, by default 1
        - "batch": int
            train param, batch size, by default 64
        - "lr": float
            train param, learning rate, by default 1e-3
        - "step": int
            train param, step_size for StepLR, by default 20
        - "gamma": float
            train param, gamma for StepLR, by default 0.5
        - "warm_up": int
            train param, number of epoch for warming up, by default 1
        - "adv": int
            train param, ratio of disc/enc training for adversarial process, by default 10
        - "device": str
            train param, 'cpu' or 'cuda', by default "cpu"
    test_items: list, defaults to None
        the raw test question list, default is None
    silent: bool, defaults to False
        whether to print processing inforamtion
    data_formation: dict, defaults to None
        Mapping "content" and "knowledge" for the item formation.
        For example, {"content": "ques_content", "knowledge": "know_name"}

    Examples
    ----------
    >>> train_data = load_items("static/test_data/disenq_train.json")[:100]
    >>> test_data = load_items("static/test_data/disenq_test.json")[:100]
    >>> tokenizer = DisenQTokenizer(max_length=250, tokenize_method="space")
    >>> train_disenqnet(train_data, tokenizer,
    ... "examples/test_model/disenq","examples/test_model/disenq", silent=True)  # doctest: +SKIP
    """
    # dataset
    default_train_params = {
        # data params
        "trim_min": 2,
        "w2v_workers": 1,
        # model params
        "hidden": 128,
        "dropout": 0.2,
        "pos_weight": 1,
        "cp": 1.5,
        "mi": 1.0,
        "dis": 2.0,
        # training params
        "epoch": 1,
        "batch": 64,
        "lr": 1e-3,
        "step": 20,
        "gamma": 0.5,
        "warm_up": 1,
        "adv": 10,
        "device": "cpu"
    }
    if train_params is not None:
        default_train_params.update(train_params)
    train_params = default_train_params

    tokenizer = DisenQTokenizer()
    if pretrained_dir:
        tokenizer = DisenQTokenizer.from_pretrained(pretrained_dir)
    else:
        tokenizer.set_vocab(train_items, key="stem")

    items = train_items + ([] if test_items is not None else test_items)
    disen_tokenizer, concept_to_idx, word2vec = preprocess_dataset(pretrained_dir, disen_tokenizer, items,
                                                                data_formation,trim_min_count=train_params["trim_min"],
                                                                embed_dim=train_params["hidden"],
                                                                w2v_params=None, silent=False)

    train_dataset = DisenQDataset(train_items, disen_tokenizer, mode="train", concept_to_idx=concept_to_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=train_params["batch"],
                                  shuffle=True, collate_fn=train_dataset.collate_fn)
    if test_items is not None:
        test_dataset = DisenQDataset(test_items, disen_tokenizer, mode="test", concept_to_idx=concept_to_idx)
        test_dataloader = DataLoader(test_dataset, batch_size=train_params["batch"],
                                  shuffle=True, collate_fn=test_dataset.collate_fn)
    else:
        test_dataloader = None

    vocab_size = len(disen_tokenizer)
    concept_size = len(concept_to_idx)
    wv = word2vec

    # model
    disen_q_net = DisenQNet(vocab_size, concept_size, train_params["hidden"], train_params["dropout"],
                            train_params["pos_weight"], train_params["cp"], train_params["mi"], train_params["dis"],
                            wv=wv, device=train_params["device"])

    disen_q_net.train(train_dataloader, test_dataloader, train_params["epoch"],
                      train_params["lr"], train_params["step"], train_params["gamma"],
                      train_params["warm_up"], train_params["adv"], silent=silent)
    disen_q_net.save_pretrained(output_dir)
