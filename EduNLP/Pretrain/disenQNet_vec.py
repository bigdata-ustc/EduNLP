from msilib.schema import Error
import os
from re import A
from cv2 import erode
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..ModelZoo.DisenQNet.DisenQNet import DisenQNet, ConceptModel
from ..Tokenizer import get_tokenizer

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
            is_num = s[:sep_index].isdigit() and s[sep_index+1:].isdigit()
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
                except:
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
    item2index = {item:index for index, item in enumerate(items)}
    return item2index


def save_list(item2index, path):
    item2index = sorted(item2index.items(), key=lambda kv: kv[1])
    items = [item for item, _ in item2index]
    with open(path, "wt", encoding="utf-8") as file:
        file.write('\n'.join(items))
    return


class DisenQTokenizer(object):
    def __init__(self, vocab_path=None, max_length=250, text_tokenzier="space", *args, **argv):
    # def __init__(self, vocab_path=None, config_path=None, max_length=250, text_tokenzier="space", *args, **argv):
        super(DisenQTokenizer, self).__init__(*args)
        
        self.text_tokenizer = get_tokenizer(text_tokenzier) if text_tokenzier != "space" else self._space_toeknzier

        self.num_token = "<num>"
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        # if config_path is not None:
        #     self.load_tokenizer_config(config_path)
        # else:
        self.max_length = max_length
        if vocab_path is not None:
            self.load_vocab(vocab_path)

    # def load_tokenizer_config(self, config_path):
    #     with open(config_path, "r", encoding="utf-8") as rf:
    #         model_config = json.load(rf)
    #         self.max_length = model_config["max_len"]

    def __call__(self, items: (list, str, dict), key=lambda x: x, padding=True,
                return_tensors=True, return_text=False, *args, **kwargs):
        token_items = self.tokenize(items, key)
        ids = list()
        lengths = list()
        for item in token_items:
            indexs = [self.word2index.get(w, self.word2index[self.unk_token]) for w in item]
            ids.append(indexs)
            lengths.append(len(indexs))
        ret = {
            "content_idxs": self.padding(ids, self.max_length) if padding else ids,
            "content_lens": lengths
        }

        if return_tensors:
            return {key: torch.as_tensor(val) for key, val in ret.items()}
        elif return_text:
            ret["content"] = token_items

        return ret

    def tokenize(self, items: (list, str, dict), key=lambda x: x, *args, **kwargs):
      if isinstance(items, str) or isinstance(items, dict):
        return self._tokenize([items], key)
      elif isinstance(items, list):
        return self._tokenize(items, key)
      else:
         raise ValueError("items should be str or list!")

    def padding(self, ids, max_length):
    #   max_len = max([len(i) for i in ids])
      padding_ids = [t + [ self.word2index[self.pad_token]] * (max_length - len(t) ) for t in ids]
      return padding_ids

    def _space_toeknzier(self, items, key=lambda x: x):
        return [key(item).strip().split(' ') for item in items]

    def _tokenize(self, items, key=lambda x: x):
        # print("[test] _tokenize",items[0])
        # print("[test] key",key(items[0]))
        items = self.text_tokenizer(items, key=key)
      
        token_items = list()
        stop_words = set("\n\r\t .,;?\"\'。．，、；？“”‘’（）")
        for text in items:
            # text = key(data).strip().split(' ')
            # print("1 ",text)
            text = [w for w in text if w != '' and w not in stop_words]
            #   print("2 ",text)
            if len(text) == 0:
                text = [self.unk_token]
            if len(text) > self.max_length:
                text = text[:self.max_length]
            text = [self.num_token if check_num(w) else w for w in text]
            #   print("3 ",text)
            token_items.append(text)
        return token_items

    def load_vocab(self, path):
        with open(path, "rt", encoding="utf-8") as file:
            self.words = file.read().strip().split('\n')
            self.word2index = {word:index for index, word in enumerate(self.words)}

    def set_vocab(self, items: list, save_path, key=lambda x: x, trim_min_count=50, silent=False):
        """
        Parameters
        -----------
        items: list
            can be the list of str, or list of dict
        save_path:
            the path to save the vocab
        key: 
            determine how to get the text of each item
        """
        word2cnt = dict()
        token_items = self._tokenize(items, key=key)
        # print("test token_items : ", token_items)
        for item in token_items:
            for w in item:
                word2cnt[w] = word2cnt.get(w, 0) + 1

        ctrl_tokens = [self.num_token, self.unk_token, self.pad_token]
        words = [w for w, c in word2cnt.items() if c >= trim_min_count and w not in ctrl_tokens]

        if not silent:
            keep_word_cnts = sum(word2cnt[w] for w in words)
            all_word_cnts = sum(word2cnt.values())
            logging.info(f"save words({trim_min_count}): {len(words)}/{len(word2cnt)} = {len(words)/len(word2cnt):.4f} with frequency {keep_word_cnts}/{all_word_cnts}={keep_word_cnts/all_word_cnts:.4f}")
        
        self.words = ctrl_tokens + sorted(words)
        
        self.word2index = {word:index for index, word in enumerate(self.words)}
        save_list(self.word2index, save_path) # only save words

    def update_vocab(self, save_path):
        save_list(self.word2index, save_path) # only save words

    @property
    def vocab_size(self):
        return len(self.word2index)


class QuestionDataset(Dataset):
    """
        Question dataset including text, length, concept Tensors
    """
    def __init__(self, items, predata_dir, max_length, dataset_type, silent=False, embed_dim=128, trim_min_count=50):
    # def __init__(self, dataset_path, max_length, dataset_type, silent, wv_path=None, embed_dim=128, trim_min_count=50, word_list_path=None, concept_list_path=None):
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

            # if not silent:
            #     logging.info(f"load vocab from {word_list_path}")
            #     logging.info(f"load concept from {concept_list_path}")
            #     logging.info(f"load word2vec from {wv_path}")
        else:
            self.disen_tokenzier = DisenQTokenizer()
            self.disen_tokenzier.set_vocab(items, self.word_list_path, key=lambda x: x["content"])
            if not silent:
                logging.info(f"save vocab to {self.word_list_path}")
            self.word2index = self.disen_tokenzier.word2index

        self.num_token = self.disen_tokenzier.num_token # "<num>"
        self.unk_token = self.disen_tokenzier.unk_token # "<unk>"
        self.pad_token = self.disen_tokenzier.pad_token # "<pad>"
        self.unk_idx = self.word2index[self.unk_token]

        # load dataset, init construct word and concept list
        if dataset_type == "train":
            self.dataset = self.process_dataset(items, trim_min_count, embed_dim, init)
        elif dataset_type == "test":
            self.dataset = self.process_dataset(items, trim_min_count, embed_dim, False)

        if not silent:
            # logging.info(f"load dataset from {dataset_path}")
            logging.info(f"processing raw data for QuestionDataset...")

        self.vocab_size = len(self.word2index)
        self.concept_size = len(self.concept2index)
        if dataset_type == "train" and not silent:
            logging.info(f"vocab size: {self.vocab_size}")
            logging.info(f"concept size: {self.concept_size}")
        return

    def process_dataset(self, items, trim_min_count, embed_dim, init=False, text_key=lambda x: x["content"]):
        # make items in standard format
        for i, item in enumerate(items):
            # print("[test] process_dataset ",items[0] )
            # print("[test] text_key ",text_key(items[0]) )
            token_data = self.disen_tokenzier(item, key=text_key, return_tensors=False, return_text=True, padding=True)
            items[i]["content_idx"] = token_data["content_idxs"][0]
            items[i]["content_len"] = token_data["content_lens"][0]
            items[i]["content"] = token_data["content"][0]

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
                self.concept2index = {concept:index for index, concept in enumerate(concepts)}
                save_list(self.concept2index, self.concept_list_path)
                if not self.silent:
                    logging.info(f"save concept to {self.concept_list_path}")
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
                wv = Word2Vec(corpus, vector_size =embed_dim, min_count=trim_min_count).wv
                # 按照 vocab 中的词序 来保存
                # wv_list = [wv[w] if w in wv.key_to_index else np.random.rand(embed_dim) for w in words]
                ctrl_tokens = [self.num_token, self.unk_token, self.pad_token]
                wv_list = [wv[w] if w not in ctrl_tokens else np.random.rand(embed_dim) for w in words ]
                self.word2vec = torch.tensor(wv_list)
                torch.save(self.word2vec, self.wv_path)
                if not self.silent:
                    logging.info(f"save word2vec to {self.wv_path}")
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
        text = [t+[pad_idx]*(max_length-len(t)) for t in text]
        text = torch.tensor(text)
        length = torch.tensor(length)
        concept = torch.tensor(concept)
        return text, length, concept


def train_disenQNet(train_items, output_dir, predata_dir, train_params=None, test_items=None):
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
    >>> tokenizer = DisenQTokenizer()
    >>> stems = ["有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$",
    ... "有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$"]
    >>> token_item = [tokenizer(i) for i in stems]
    >>> print(token_item[0].keys())
    dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
    >>> finetune_bert(token_item, "examples/test_model/data/bert") #doctest: +ELLIPSIS
    {'train_runtime': ..., ..., 'epoch': 1.0}
    """
    # dataset
    default_train_params = {
        "trim_min" : 50,
        "max_len" : 250,

        "hidden" : 128,
        "dropout" : 0.2,
        "pos_weight" : 1,

        "cp" : 1.5,
        "mi" : 1.0,
        "dis" : 2.0,

        "epoch" : 10,
        "batch" : 128,
        "lr" : 1e-3,
        "step" : 20,
        "gamma" : 0.5,
        "warm_up" : 5,
        "adv" : 10,

        "device": "cuda",
    }

    if train_params is not None:
        default_train_params.update(train_params)
    train_params = default_train_params


    # wv_path = os.path.join(data_path, "wv.th")
    # word_path = os.path.join(data_path, "vocab.list")
    # concept_path = os.path.join(data_path, "concept.list")
    # use_predata_dir = True if os.path.exists(predata_dir) else False


    # train_items = load_data(os.path.join(data_path, "train.json")) # can replay by items
    # assert train_items is not None
    train_dataset = QuestionDataset(train_items, predata_dir, train_params["max_len"], "train", silent=False,
                                    embed_dim=train_params["hidden"], trim_min_count=train_params["trim_min"])
    train_dataloader = DataLoader(train_dataset, batch_size=train_params["batch"],
                                    shuffle=True, collate_fn=train_dataset.collate_data)

    # test_items = load_data(os.path.join(data_path, "test.json"))
    if test_items is not None:
        test_dataset = QuestionDataset(test_items, predata_dir, train_params["max_len"], "test", silent=False,
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
                        train_params["warm_up"], train_params["adv"], silent=False)
    disen_q_net_path = os.path.join(output_dir, "disen_q_net.th")
    disen_q_net.save(disen_q_net_path)
