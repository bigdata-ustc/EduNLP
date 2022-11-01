"""Pre-process input text, tokenizing, building vocabs, and pre-train word
level vectors."""

import os
import logging
from pickle import NONE
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import signal
import threading
from tqdm import tqdm
from functools import partial
from collections import namedtuple
from copy import copy
import json
import math
import queue
import random
from PIL import Image
from torchvision.transforms.functional import to_grayscale
from torchvision.transforms.functional import to_tensor
from gensim.models import Word2Vec
from ..SIF.segment.segment import FigureSegment
from ..SIF.segment import seg
from ..SIF.tokenization import tokenize
from ..SIF import EDU_SPYMBOLS
from ..ModelZoo.quesnet import QuesNetForPreTraining, AE
from .pretrian_utils import PretrainedEduTokenizer
from EduNLP import logger
import linecache
from typing import List, Union, Optional

Question = namedtuple('Question',
                      ['id', 'content', 'answer', 'false_options', 'labels'])


def save_list(item2index, path):
    item2index = sorted(item2index.items(), key=lambda kv: kv[1])
    items = [item for item, _ in item2index]
    with open(path, "wt", encoding="utf-8") as file:
        file.write('\n'.join(items))
    return


class QuesNetTokenizer(PretrainedEduTokenizer):
    """
    Examples
    --------
    >>> tokenizer = QuesNetTokenizer(meta=['knowledge'])
    >>> test_items = [{"ques_content": "$\\triangle A B C$ 的内角为 $A, \\quad B, $\\FigureID{test_id}$",
    ... "knowledge": "['*', '-', '/']"}, {"ques_content": "$\\triangle A B C$ 的内角为 $A, \\quad B",
    ... "knowledge": "['*', '-', '/']"}]
    >>> tokenizer.set_vocab(test_items,
    ... trim_min_count=1, key=lambda x: x["ques_content"], silent=True)
    >>> tokenizer.set_meta_vocab(test_items, silent=True)
    >>> token_items = [tokenizer(i, key=lambda x: x["ques_content"]) for i in test_items]
    >>> print(token_items[0].keys())
    dict_keys(['seq_idx', 'meta_idx'])
    >>> token_items = tokenizer(test_items, key=lambda x: x["ques_content"])
    >>> print(len(token_items["seq_idx"]))
    2
    """

    def __init__(self, vocab_path=None, meta_vocab_dir=None, img_dir: str = None,
                 max_length=250, tokenize_method="custom", symbol="mas", add_specials: list = None,
                 meta: List[str] = None, img_token='<img>', unk_token="<unk>", pad_token="<pad>", **argv):
        """
        Parameters
        ----------
        img_dir : str, optional
            path of images, by default None
        vocab_path : str, optional
            path of vacab file, by default None
        max_length : int, optional
            by default 250
        meta : list, optional
            the name of meta (side information), by default 'knowledge'
        img_token : str, optional
            by default '<img>'
        unk_token : str, optional
            by default "<unk>"
        pad_token : str, optional
            by default "<pad>"
        """
        if add_specials is None:
            add_specials = [img_token]
        else:
            add_specials = [img_token] + add_specials
        self.tokenization_params = {
            "formula_params": {
                "method": "linear",
                "skip_figure_formula": True
            }
        }
        argv.update(self.tokenization_params)
        super().__init__(vocab_path=vocab_path, max_length=max_length, tokenize_method=tokenize_method,
                         add_specials=add_specials, unk_token=unk_token, pad_token=pad_token, symbol=symbol, **argv)
        if meta is None:
            meta = ['know_name']
        self.img_dir = img_dir
        self.img_token = img_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.meta = meta
        self.meta_vocab_dir = meta_vocab_dir

        self.stoi = dict()
        self.itos = dict()
        self.stoi['word'] = self.vocab.token_to_idx
        self.itos['word'] = self.vocab.idx_to_token
        if meta_vocab_dir is not None:
            self.load_meta_vocab(meta_vocab_dir=meta_vocab_dir)
        self.config = {
            k: v for k, v in locals().items() if k not in [
                "self", "__class__", "vocab_path", 'img_dir', 'meta_vocab_dir']
        }

    def __call__(self, item: Union[str, dict, list], key=lambda x: x,
                 meta: Optional[list] = None,
                 padding=False, return_text=False, *args, **kwargs):
        """
        item: str or dict
            the question item
        key: function
            determine how to get the text of each item
        padding: bool
            whether to pad the seq_idx
        return_text: bool
            whether to return text tokens
        """
        if isinstance(item, list):
            ret = {
                "seq_idx": [],
                "meta_idx": []
            }
            if return_text:
                ret["seq_token"] = []
                ret["meta"] = []
            for i in item:
                r = self._convert_to_ids(i, key, meta, padding, return_text, *args, **kwargs)
                ret["seq_idx"].append(r["seq_idx"])
                ret["meta_idx"].append(r["meta_idx"])
                if return_text:
                    ret["seq_token"].append(r["seq_token"])
                    ret["meta"].append(r["meta"])
        else:
            ret = self._convert_to_ids(item, key, meta, padding, return_text, *args, **kwargs)

        return ret

    def _convert_to_ids(self, item: Union[str, dict, list], key=lambda x: x,
                        meta: Optional[list] = None,
                        padding=False, return_text=False, *args, **kwargs):
        token_item = self.tokenize(item, key)
        token_idx = []
        for _, w in enumerate(token_item):
            if isinstance(w, FigureSegment):
                # image
                try:
                    im = Image.open(os.path.join(self.img_dir, f'{w.src[10:-1]}.png'))
                    im = im.resize((56, 56))
                    token_idx.append(to_grayscale(im))
                except Exception:
                    warnings.warn('Open image error!')
                    token_idx.append(self.stoi['word'][self.img_token])
            else:
                # word
                token_idx.append(self.stoi['word'].get(w) or self.stoi['word'][self.unk_token])

        meta_idxs = {}
        meta_items = {}
        if meta is None:
            meta = self.meta
        for m in meta:
            meta_idx = []
            if isinstance(item, dict) and m in item:
                meta_item = item[m]
                if isinstance(meta_item, str):
                    if meta_item.startswith('['):
                        # a list of labels ['+', '-', '/']
                        meta_item = eval(meta_item)
                elif isinstance(meta_item, list):
                    pass
                else:
                    raise Exception("Side information must be a list!")
                meta_items[m] = meta_item
                for k in meta_item:
                    meta_idx.append(self.stoi[m].get(k) or self.stoi[m][self.unk_token])
            meta_idxs[m] = meta_idx
        ret = {
            "seq_idx": self.padding(token_idx, self.max_length) if padding else token_idx,
            "meta_idx": meta_idxs
        }

        if return_text:
            ret["seq_token"] = token_item
            ret["meta"] = meta_items
        return ret

    def load_meta_vocab(self, meta_vocab_dir):
        for m in self.meta:
            try:
                with open(os.path.join(meta_vocab_dir, f'meta_{m}.txt'), "rt", encoding="utf-8") as f:
                    meta = f.read().strip().split('\n')
                    self.stoi[m] = {word: index for index, word in enumerate(meta)}
                    self.itos[m] = {i: s for s, i in self.stoi[m].items()}
            except Exception:
                self.stoi[m] = None
                self.itos[m] = None

    def set_meta_vocab(self, items: list, meta: List[str] = None, silent=True):
        self.meta = meta if meta is not None else self.meta
        for m in self.meta:
            meta = set()
            if m in items[0]:
                for item in items:
                    meta_item = item[m]
                    if isinstance(meta_item, str):
                        if meta_item.startswith('['):
                            # a list of labels ['+', '-', '/']
                            meta_item = eval(meta_item)
                    elif isinstance(meta_item, list):
                        pass
                    else:
                        raise Exception("Side information must be a list!")
                    meta = meta | set(meta_item)
            meta = [self.unk_token] + sorted(meta)
            if not silent:
                print(f"save meta information {m}: {len(meta)}")
            self.stoi[m] = {word: index for index, word in enumerate(meta)}
            self.itos[m] = {i: s for s, i in self.stoi[m].items()}

    def set_vocab(self, items: list, key=lambda x: x, lower: bool = False,
                  trim_min_count: int = 1, do_tokenize: bool = True, silent=True):
        """
        Parameters
        -----------
        items: list
            can be the list of str, or list of dict
        key: function
            determine how to get the text of each item
        trim_min_count
        silent
        """
        token_items = self.tokenize(items, key) if do_tokenize else [key(item) for item in items]
        self.vocab.set_vocab(corpus_items=token_items, trim_min_count=trim_min_count, lower=lower, silent=silent)
        self.stoi['word'] = self.vocab.token_to_idx
        self.itos['word'] = self.vocab.idx_to_token

    @classmethod
    def from_pretrained(cls, tokenizer_config_dir, img_dir=None, **argv):
        """
        Parameters:
        -----------
        tokenizer_config_dir: str
            must contain tokenizer_config.json and vocab.txt and meta_{meta_name}.txt
        img_dir: str
            default None
            the path of image directory
        """
        tokenizer_config_path = os.path.join(tokenizer_config_dir, "tokenizer_config.json")
        pretrained_vocab_path = os.path.join(tokenizer_config_dir, "vocab.txt")
        with open(tokenizer_config_path, "r", encoding="utf-8") as rf:
            tokenizer_config = json.load(rf)
            tokenizer_config.update(argv)
            return cls(
                vocab_path=pretrained_vocab_path,
                meta_vocab_dir=tokenizer_config_dir,
                img_dir=img_dir,
                max_length=tokenizer_config["max_length"],
                tokenize_method=tokenizer_config["tokenize_method"],
                meta=tokenizer_config["meta"],
                img_token=tokenizer_config["img_token"],
                unk_token=tokenizer_config["unk_token"],
                pad_token=tokenizer_config["pad_token"])

    def save_pretrained(self, tokenizer_config_dir):
        """Save tokenizer into local files

        Parameters:
        -----------
        tokenizer_config_dir: str
            save tokenizer params in `/tokenizer_config.json` and save words in `vocab.txt`
            and save metas in `meta_{meta_name}.txt`
        """
        if not os.path.exists(tokenizer_config_dir):
            os.makedirs(tokenizer_config_dir, exist_ok=True)
        tokenizer_config_path = os.path.join(tokenizer_config_dir, "tokenizer_config.json")
        self.vocab.save_vocab(os.path.join(tokenizer_config_dir, "vocab.txt"))
        for m in self.meta:
            save_list(self.stoi[m], os.path.join(tokenizer_config_dir, f'meta_{m}.txt'))
        with open(tokenizer_config_path, "w", encoding="utf-8") as wf:
            json.dump(self.config, wf, ensure_ascii=False, indent=2)

    def padding(self, idx, max_length, type='word'):
        padding_idx = idx + [self.stoi[type][self.pad_token]] * (max_length - len(idx))
        return padding_idx

    def set_img_dir(self, path):
        self.img_dir = path


def clip(v, low, high):
    if v < low:
        v = low
    if v > high:
        v = high
    return v


class Lines:
    def __init__(self, filename, skip=0, preserve_newline=False):
        self.filename = filename
        with open(filename, "r", encoding="utf-8") as f:
            self.length = len(f.readlines()) - skip
        assert self.length > 0, f'{filename} is empty. Or file length is less than skip length.'
        self.skip = skip
        self.preserve_newline = preserve_newline

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        d = self.skip + 1
        if isinstance(item, int):
            if item < self.length:
                line = linecache.getline(self.filename,
                                         item % len(self) + d)
                if self.preserve_newline:
                    return json.loads(line)
                else:
                    return json.loads(line.strip('\r\n'))

        elif isinstance(item, slice):
            low = 0 if item.start is None else item.start
            low = clip(low, -len(self), len(self) - 1)
            if low < 0:
                low += len(self)
            high = len(self) if item.stop is None else item.stop
            high = clip(high, -len(self), len(self))
            if high < 0:
                high += len(self)
            ls = []
            for i in range(low, high):
                line = linecache.getline(self.filename, i + d)
                if not self.preserve_newline:
                    line = line.strip('\r\n')
                ls.append(json.loads(line))

            return ls

        raise IndexError('index must be int or slice')


class QuestionLoader:
    def __init__(self, ques: Lines, tokenizer: QuesNetTokenizer,
                 pipeline=None, range=None, meta: Optional[list] = None,
                 content_key=lambda x: x['ques_content'],
                 meta_key=lambda x: x['know_name'],
                 answer_key=lambda x: x['ques_answer'],
                 option_key=lambda x: x['ques_options'],
                 skip=0
                 ):
        """ Read question file as data list. Same behavior on same file.

        Parameters
        ----------
        ques_file : str
            path of question file
        tokenizer : QuesNetTokenizer
        pipeline : _type_, optional
            _description_, by default None
        range : _type_, optional
            _description_, by default None
        content_key : function, optional
            by default lambda x:x['ques_content']
        meta_key : function, optional
            by default lambda x:x['know_name']
        answer_key: function, optional
            by default lambda x:x['ques_answer']
        option_key: function, optional
            by default lambda x:x['ques_options']
        skip: int, optional
            skip the first several lines, by default 0
        """
        self.range = None
        self.ques = ques
        self.range = range or slice(0, len(self), skip)
        self.img_dir = tokenizer.img_dir
        self.labels = []
        self.stoi = tokenizer.stoi
        self.tokenizer = tokenizer

        self.content_key = content_key
        self.meta = meta if meta else tokenizer.meta
        self.meta_key = meta_key
        self.answer_key = answer_key
        self.option_key = option_key

        self.pipeline = pipeline

    def split_(self, split_ratio):
        first_size = int(len(self) * (1 - split_ratio))
        other = copy(self)
        self.range = slice(0, first_size, 1)
        other.range = slice(first_size, len(other), 1)
        return other

    def __len__(self):
        return len(self.ques) if self.range is None \
            else self.range.stop - self.range.start

    def __getitem__(self, x):
        if isinstance(x, int):
            x += self.range.start
            item = slice(x, x + 1, 1)
        else:
            item = slice(x.start + self.range.start,
                         x.stop + self.range.start, 1)
        qs = []
        if item.start > len(self):
            raise IndexError
        for line in self.ques[item]:
            q = line
            qid = q['ques_id']
            token = self.tokenizer(q, key=self.content_key, meta=self.meta)
            content = token['seq_idx']
            meta = token['meta_idx']
            if self.answer_key(q).isalpha() and len(self.answer_key(q)) == 1 and ord(self.answer_key(q)) < 128 and len(
                    self.option_key(q)) > 0:
                answer_idx = ord(self.answer_key(q).upper()) - ord('A')
                options = self.option_key(q)
                answer = self.tokenizer(options.pop(answer_idx), meta=self.meta)
                answer = answer['seq_idx']
                false_options = [(self.tokenizer(option, meta=self.meta))['seq_idx'] for option in options]
                qs.append(Question(qid, content, answer, false_options, meta))
            else:
                answer = (self.tokenizer(self.answer_key(q), meta=self.meta))['seq_idx']
                qs.append(Question(qid, content, answer, [[0], [0], [0]], meta))

        if callable(self.pipeline):
            qs = self.pipeline(qs)
        if isinstance(x, int):
            return qs[0]
        else:
            return qs


def optimizer(*models, **kwargs):
    _cur_optim = [m.optim_cls(m.parameters(), **kwargs)
                  if hasattr(m, 'optim_cls')
                  else torch.optim.Adam(m.parameters(), **kwargs)
                  for m in models]
    if len(_cur_optim) == 1:
        return _cur_optim[0]
    else:
        return _cur_optim


class PrefetchIter:
    """Iterator on data and labels, with states for save and restore."""

    def __init__(self, data, *label, length=None, batch_size=1, shuffle=True):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=8)
        self.length = length if length is not None else len(data)

        assert all(self.length == len(lab) for lab in label), \
            'data and label must have same lengths'

        self.index = list(range(len(self)))
        if shuffle:
            random.shuffle(self.index)
        self.thread = None
        self.pos = 0

    def __len__(self):
        return math.ceil(self.length / self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.produce, daemon=True)
            self.thread.start()

        if self.pos >= len(self.index):
            raise StopIteration

        item = self.queue.get()
        if isinstance(item, Exception):
            raise item
        else:
            self.pos += 1
            return item

    def produce(self):
        for i in range(self.pos, len(self.index)):
            try:
                index = self.index[i]

                bs = self.batch_size

                if callable(self.data):
                    data_batch = self.data(index * bs, (index + 1) * bs)
                else:
                    data_batch = self.data[index * bs:(index + 1) * bs]

                label_batch = [label[index * bs:(index + 1) * bs]
                               for label in self.label]
                if label_batch:
                    self.queue.put([data_batch] + label_batch)
                else:
                    self.queue.put(data_batch)
            except Exception as e:
                self.queue.put(e)
                return


class EmbeddingDataset(Dataset):
    def __init__(self, data, data_type='image'):
        self.data = data
        self.data_type = data_type
        assert self.data_type in ['image', 'meta']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data_type == 'image':
            return to_tensor(self.data[idx])
        elif self.data_type == 'meta':
            return self.data[idx]


def pretrain_iter(ques, batch_size):
    _cur_iter = PrefetchIter(ques, batch_size=batch_size)
    return _cur_iter


sigint_handler = signal.getsignal(signal.SIGINT)


def critical(f):
    it = iter(f)
    signal_received = ()

    def handler(sig, frame):
        nonlocal signal_received
        signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    while True:
        try:
            signal.signal(signal.SIGINT, handler)
            yield next(it)
            signal.signal(signal.SIGINT, sigint_handler)
            if signal_received:
                sigint_handler(*signal_received)
        except StopIteration:
            break


def pretrain_embedding_layer(dataset: EmbeddingDataset, ae: AE, lr: float = 1e-3, log_step: int = 1, epochs: int = 3,
                             batch_size: int = 4, device=torch.device('cpu')):
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(ae.parameters(), lr=lr)
    ae.train()
    ae.to(device)
    train_type = dataset.data_type
    for i in range(epochs):
        for batch, item in tqdm(enumerate(train_dataloader)):
            loss = ae.loss(item.to(device))
            optim.zero_grad()
            loss.backward()
            optim.step()
            if batch % log_step == 0:
                logger.info(f"[Epoch{i}][Batch{batch}]Training {train_type} Embedding layer, loss:{loss}")
    return ae


def pretrain_quesnet(path, output_dir, img_dir=None, save_embs=False, train_params=None):
    """ pretrain quesnet

    Parameters
    ----------
    path : str
        path of question file
    output_dir : str
        output path·
    tokenizer : QuesNetTokenizer
        quesnet tokenizer
    save_embs : bool, optional
        whether to save pretrained word/image/meta embeddings seperately
    train_params : dict, optional
        the training parameters and model parameters, by default None
        - "n_epochs": int, default = 1
            train param, number of epochs
        - "batch_size": int, default = 6
            train param, batch size
        - "lr": float, default = 1e-3
            train param, learning rate
        - "save_every": int, default = 0
            train param, save steps interval
        - "log_steps": int, default = 10
            train param, log steps interval
        - "device": str, default = 'cpu'
            train param, 'cpu' or 'cuda'
        - "max_steps": int, default = 0
            train param, stop training when reach max steps
        - "emb_size": int, default = 256
            model param, the embedding size of word, figure, meta info
        - "feat_size": int, default = 256
            model param, the size of question infer vector

    Examples
    ----------
    >>> tokenizer = QuesNetTokenizer(meta=['know_name'])
    >>> items = [{"ques_content": "若复数$z=1+2 i+i^{3}$，则$|z|=$，$\\FigureID{000004d6-0479-11ec-829b-797d5eb43535}$",
    ... "ques_id": "726cdbec-33a9-11ec-909c-98fa9b625adb",
    ... "know_name": "['代数', '集合', '集合的相等']"
    ... }]
    >>> tokenizer.set_vocab(items, key=lambda x: x['ques_content'], trim_min_count=1, silent=True)
    >>> pretrain_quesnet('./data/quesnet_data.json', './testQuesNet', tokenizer) # doctest: +SKIP
    """
    default_train_params = {
        # train params
        "n_epochs": 1,
        "batch_size": 8,
        "lr": 1e-3,
        'save_every': 0,
        'log_steps': 10,
        'device': 'cpu',
        'max_steps': 0,
        # model params
        'emb_size': 256,
        'feat_size': 256,
    }
    if train_params is not None:
        default_train_params.update(train_params)
    train_params = default_train_params
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(train_params['device'])
    # set tokenizer
    tokenizer = QuesNetTokenizer(meta=['know_name'],
                                 img_dir=img_dir)
    items = Lines(path, skip=1)

    tokenizer.set_vocab(items, key=lambda x: x['ques_content'],
                        trim_min_count=2, silent=False)
    tokenizer.set_meta_vocab(items, silent=False)
    tokenizer.save_pretrained(output_dir)

    ques_dl = QuestionLoader(items, tokenizer)
    model = QuesNetForPreTraining(_stoi=tokenizer.stoi, feat_size=train_params['feat_size'],
                                  emb_size=train_params['emb_size']).to(device)
    emb_dict = tokenizer.stoi['word']
    emb_dict_rev = tokenizer.itos['word']
    emb_size = train_params['emb_size']
    meta_size = model.quesnet.meta_size
    w2v_corpus = []
    img_corpus = []
    meta_corpus = []
    for i, qs in enumerate(tqdm(ques_dl)):
        text_content = []
        for c in qs.content:
            if isinstance(c, int):
                text_content.append(emb_dict_rev[c])
            else:
                img_corpus.append(c)
        for a in qs.answer:
            if isinstance(a, int):
                text_content.append(emb_dict_rev[a])
            else:
                img_corpus.append(a)
        w2v_corpus.append(text_content)
        meta_vector = torch.zeros(meta_size, dtype=torch.float)
        for m in qs.labels[model.quesnet.meta]:
            meta_vector.add_(torch.nn.functional.one_hot(torch.tensor(m, dtype=torch.int64),
                             model.quesnet.meta_size).to(torch.float))
        meta_corpus.append(meta_vector)

    # train word2vec for text embedding
    gensim_w2v = Word2Vec(sentences=[[item] for item in emb_dict.keys()], min_count=1,
                          vector_size=emb_size)
    gensim_w2v.init_weights()
    gensim_w2v.train(corpus_iterable=w2v_corpus, total_examples=len(w2v_corpus), epochs=train_params['n_epochs'])
    w2v_emb = gensim_w2v.syn1neg
    emb_weights = []
    for key, item in emb_dict.items():
        w2v_index = gensim_w2v.wv.key_to_index[key]
        emb_weights.append(w2v_emb[w2v_index])
    emb_weights = np.array(emb_weights)
    model.quesnet.load_emb(emb_weights)
    logger.info('quesnet Word Embedding loaded')
    if save_embs:
        np.save(os.path.join(output_dir, 'w2v_embs.npy'), emb_weights)

    # train auto-encoder loss for image embedding
    img_dataset = EmbeddingDataset(data=img_corpus, data_type='image')
    trained_ie = pretrain_embedding_layer(dataset=img_dataset, ae=model.quesnet.ie, lr=train_params['lr'],
                                          log_step=train_params['log_steps'], batch_size=train_params['batch_size'],
                                          epochs=train_params['n_epochs'], device=device)
    model.quesnet.load_img(trained_ie)
    logger.info('quesnet Image Embedding loaded')
    if save_embs:
        torch.save(trained_ie.state_dict(), os.path.join(output_dir, 'trained_ie.pt'))

    # train auto-encoder loss for meta embedding
    meta_dateset = EmbeddingDataset(data=meta_corpus, data_type='meta')
    trained_me = pretrain_embedding_layer(dataset=meta_dateset, ae=model.quesnet.me, lr=train_params['lr'],
                                          log_step=train_params['log_steps'], batch_size=train_params['batch_size'],
                                          epochs=train_params['n_epochs'], device=device)
    model.quesnet.load_meta(trained_me)
    logger.info('quesnet Meta Embedding loaded')
    if save_embs:
        torch.save(trained_me.state_dict(), os.path.join(output_dir, 'trained_me.pt'))

    logger.info("quesnet Word, Image and Meta Embeddings training is done")

    # HLM and DOO training
    ques_dl.pipeline = partial(model.quesnet.make_batch, device=device, pretrain=True)
    model.train()
    optim = optimizer(model, lr=train_params['lr'])
    n_batches = 0
    for epoch in range(0, train_params['n_epochs']):
        train_iter = pretrain_iter(ques_dl, train_params['batch_size'])
        bar = enumerate(tqdm(train_iter, initial=train_iter.pos),
                        train_iter.pos)
        for i, batch in critical(bar):
            n_batches += 1
            loss = model(batch).loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            if train_params['save_every'] > 0 and i % train_params['save_every'] == 0:
                # model.save(os.path.join(output_dir, f'QuesNet_{epoch}.{i}'))
                model.save_pretrained(output_dir)

            if train_params['log_steps'] > 0 and i % train_params['log_steps'] == 0:
                logger.info(f"{epoch}.{i}---loss: {loss.item()}")

            if train_params['max_steps'] > 0 and n_batches % train_params['max_steps'] == 0:
                break

        # model.save(os.path.join(output_dir, f'QuesNet_{epoch}'))

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
