"""Pre-process input text, tokenizing, building vocabs, and pre-train word
level vectors."""

import os
import logging
import torch
import signal
import threading
from tqdm import tqdm
from functools import partial
from collections import namedtuple
from pathlib import Path
from copy import copy
import json
import math
import queue
import random
from typing import Union, Optional
from types import FunctionType as function

import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_grayscale
from ..SIF.segment.segment import FigureSegment

from ..SIF.segment import seg
from ..SIF.tokenization import tokenize
from ..ModelZoo.QuesNet import QuesNet

import linecache
import subprocess

Question = namedtuple('Question',
                      ['id', 'content', 'answer', 'false_options', 'labels'])


def save_list(item2index, path):
    item2index = sorted(item2index.items(), key=lambda kv: kv[1])
    items = [item for item, _ in item2index]
    with open(path, "wt", encoding="utf-8") as file:
        file.write('\n'.join(items))
    return


class QuesNetTokenizer(object):
    """
    Examples
    --------
    >>> tokenizer = QuesNetTokenizer()
    >>> test_items = [{
    ...     "ques_id": "946",
    ...     "ques_content": "$\\triangle A B C$ 的内角为 $A, \\quad B, $\FigureID{test_id}$",
    ...     "knowledge": "['*', '-', '/']"}]
    >>> tokenizer.set_vocab(test_items,
    ...     trim_min_count=1, key=lambda x: x["content"], silent=True)
    >>> token_items = [tokenizer(i, key=lambda x: x["content"]) for i in test_items]
    >>> print(token_items[0].keys())
    dict_keys(['content_idx', 'meta_idx'])
    """
    def __init__(self, img_dir=None, vocab_path=None, max_length=250, meta=['know_name'],
                 img_token='<img>', unk_token="<unk>", pad_token="<pad>", *args, **argv):
        """
        Parameters
        ----------
        img_dir : str, optional
            path of images, by default None
        vocab_path : str, optional
            path of vacab file, by default None
        max_length : int, optional
            by default 250
        meta : str, optional
            the name of meta (side information), by default 'knowledge'
        img_token : str, optional
            by default '<img>'
        unk_token : str, optional
            by default "<unk>"
        pad_token : str, optional
            by default "<pad>"
        """

        self.img_dir = img_dir
        self.img_token = img_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.meta = meta

        self.stoi = dict()
        self.itos = dict()
        if vocab_path is not None:
            self.load_vocab(vocab_path)
            self.secure = True
        else:
            self.secure = False

        self.tokenization_params = {
            "formula_params": {
                "method": "linear",
                "skip_figure_formula": True
            }
        }

    def tokenize(self, item: Union[str, dict], key=lambda x: x, *args, **kwargs):
        if not self.secure:
            raise Exception("Must set the vocab first before tokenize item (either set_vocab() or load_vocab() )")

        token_text = tokenize(seg(key(item), symbol="mas"), **self.tokenization_params).tokens
        if len(token_text) == 0:
            token_text = [self.unk_token]
        if len(token_text) > self.max_length:
            token_text = token_text[:self.max_length]
        # img保留为\FigureID{}的格式，在__call__中处理为图片
        return token_text

    def __call__(self, item: Union[str, dict], key=lambda x: x,
                 meta: Optional[list] = None,
                 padding=False, return_text=False, *args, **kwargs):
        """
        item: str or dict
            the question item
        key: function
            determine how to get the text of each item
        padding: bool
            whether to pad the content_idx
        return_text: bool
            whether to return text tokens
        """
        token_item = self.tokenize(item, key)
        token_idx = []
        for _, w in enumerate(token_item):
            if isinstance(w, FigureSegment):
                # image
                try:
                    im = Image.open(os.path.join(self.img_dir, w.src[10:-1]))
                    im = im.resize((56, 56))
                    token_idx.append(to_grayscale(im))
                except Exception:
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
            "content_idx": self.padding(token_idx, self.max_length) if padding else token_idx,
            "meta_idx": meta_idxs
        }

        if return_text:
            ret["content"] = token_item
            ret["meta"] = meta_items

        return ret

    def load_vocab(self, path):
        """

        Parameters
        ----------
        path : str
            path of vocabulary files
            it must be a directory containing word.txt (meta.txt is optional)
        """
        with open(os.path.join(path, 'word.txt'), "rt", encoding="utf-8") as f:
            words = f.read().strip().split('\n')
            self.stoi['word'] = {word: index for index, word in enumerate(words)}
            self.itos['word'] = {i: s for i, s in self.stoi['word'].items()}
        for m in self.meta:
            try:
                with open(os.path.join(path, f'meta_{m}.txt'), "rt", encoding="utf-8") as f:
                    meta = f.read().strip().split('\n')
                    self.stoi[m] = {word: index for index, word in enumerate(meta)}
                    self.itos[m] = {i: s for i, s in self.stoi[m].items()}
            except Exception:
                self.stoi[m] = None
                self.itos[m] = None

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
        # word
        word2cnt = dict()
        for item in items:
            token_item = self.tokenize(item, key=key)
            for w in token_item:
                if not isinstance(w, FigureSegment):
                    word2cnt[w] = word2cnt.get(w, 0) + 1
        ctrl_tokens = [self.unk_token, self.img_token, self.pad_token]
        words = [w for w, c in word2cnt.items() if c >= trim_min_count and w not in ctrl_tokens]
        if not silent:
            keep_word_cnts = sum(word2cnt[w] for w in words)
            all_word_cnts = sum(word2cnt.values())
            print(f"save words({trim_min_count}): {len(words)}/{len(word2cnt)} = {len(words)/len(word2cnt):.4f}\
                  with frequency {keep_word_cnts}/{all_word_cnts}={keep_word_cnts/all_word_cnts:.4f}")

        vocab = ctrl_tokens + sorted(words)
        self.stoi['word'] = {word: index for index, word in enumerate(vocab)}
        self.itos['word'] = {i: s for i, s in self.stoi['word'].items()}

        # meta
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
            self.itos[m] = {i: s for i, s in self.stoi[m].items()}

    def save_vocab(self, save_vocab_path):
        """

        Parameters
        ----------
        save_vocab_path : str
            path to save word vocabulary and meta vocabulary
        """
        save_list(self.stoi['word'], os.path.join(save_vocab_path, 'word.txt'))
        for m in self.meta:
            save_list(self.stoi[m], os.path.join(save_vocab_path, f'meta_{m}.txt'))

    @classmethod
    def from_pretrained(cls, tokenizer_config_dir, img_dir=None):
        """
        Parameters:
        -----------
        tokenizer_config_dir: str
            must contain tokenizer_config.json and vocab/word.txt vocab/meta_{meta_name}.txt
        img_dir: str
            default None
            the path of image directory
        """
        tokenizer_config_path = os.path.join(tokenizer_config_dir, "tokenizer_config.json")
        pretrained_vocab_path = os.path.join(tokenizer_config_dir, "vocab")
        with open(tokenizer_config_path, "r", encoding="utf-8") as rf:
            tokenizer_config = json.load(rf)
            return cls(
                vocab_path=pretrained_vocab_path, max_length=tokenizer_config["max_length"],
                img_token=tokenizer_config["img_token"], unk_token=tokenizer_config["unk_token"],
                pad_token=tokenizer_config["pad_token"], meta=tokenizer_config["meta"], img_dir=img_dir)

    def save_pretrained(self, tokenizer_config_dir):
        """
        Parameters:
        -----------
        tokenizer_config_dir: str
            save tokenizer params in tokenizer_config.json and save words in vocab.list
        """
        tokenizer_config_path = os.path.join(tokenizer_config_dir, "tokenizer_config.json")
        save_vocab_path = os.path.join(tokenizer_config_dir, "vocab")
        os.makedirs(save_vocab_path, exist_ok=True)
        tokenizer_params = {
            "img_token": self.img_token,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "max_length": self.max_length,
            "meta": self.meta
        }
        self.save_vocab(save_vocab_path)
        with open(tokenizer_config_path, "w", encoding="utf-8") as wf:
            json.dump(tokenizer_params, wf, ensure_ascii=False, indent=2)

    def padding(self, idx, max_length, type='word'):
        padding_idx = idx + [self.stoi[type][self.pad_token]] * (max_length - len(idx))
        return padding_idx

    @property
    def vocab_size(self):
        return len(self.stoi['word'])

    def set_img_dir(self, path):
        self.img_dir = path


def clip(v, low, high):
    if v < low:
        v = low
    if v > high:
        v = high
    return v


class lines:
    def __init__(self, filename, skip=0, preserve_newline=False):
        self.filename = filename
        with open(filename):
            pass
        output = subprocess.check_output(('wc -l ' + filename).split())
        self.length = int(output.split()[0]) - skip
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
                    return line
                else:
                    return line.strip('\r\n')

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
                ls.append(line)

            return ls

        raise IndexError('index must be int or slice')


class QuestionLoader:
    def __init__(self, ques_file, tokenizer: QuesNetTokenizer,
                 pipeline=None, range=None, meta: Optional[list]=None,
                 content_key=lambda x: x['ques_content']):
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
            by default lambdax:x['ques_content']
        meta_key : function, optional
            by default lambdax:x['know_name']
        """
        self.range = None
        self.ques = lines(ques_file, skip=1)
        self.range = range or slice(0, len(self), 1)
        self.img_dir = tokenizer.img_dir
        self.labels = []
        self.stoi = tokenizer.stoi
        self.tokenizer = tokenizer

        self.content_key = content_key
        self.meta = meta if meta else tokenizer.meta

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

        for line in self.ques[item]:
            q = json.loads(line)
            qid = q['ques_id']
            token = self.tokenizer(q, key=self.content_key, meta=self.meta)
            content = token['content_idx']
            meta = token['meta_idx']
            # TODO: answer
            # TODO: false_options
            qs.append(Question(qid, content, [0], [0], meta))

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


def save_model(model, path, model_name, tag):
    path = os.path.join(path, f'{model_name}_{tag}.pt')
    path = Path(path)
    torch.save(model.state_dict(), path.open('wb'))


def pretrain_QuesNet(path, output_dir, tokenizer, train_params=None):
    default_train_params = {
        "n_epochs": 1,
        "batch_size": 4,
        "lr": 1e-3,
        'save_every': 1,
        'log_steps': 1
        # TODO: more params
    }
    if train_params is not None:
        default_train_params.update(train_params)
    train_params = default_train_params

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ques_dl = QuestionLoader(path, tokenizer)
    model = QuesNet(_stoi=tokenizer.stoi).to(device)

    # TODO: pretrain embedding layers: MetaAE, ImageAE, word2vec

    ques_dl.pipeline = partial(model.make_batch, device=device, pretrain=True)

    model.train()
    optim = optimizer(model, lr=train_params['lr'])
    n_batches = 0
    print('begin')
    for epoch in range(0, train_params['n_epochs']):
        train_iter = pretrain_iter(ques_dl, train_params['batch_size'])

        try:
            bar = enumerate(tqdm(train_iter, initial=train_iter.pos),
                            train_iter.pos)
            for i, batch in critical(bar):
                n_batches += 1
                loss = model.pretrain_loss(batch)
                if isinstance(loss, dict):
                    total_loss = 0.
                    for k, v in loss.items():
                        if v is not None:
                            total_loss += v
                    loss = total_loss
                optim.zero_grad()
                loss.backward()
                optim.step()

                if train_params['save_every'] > 0 and i % train_params['save_every'] == 0:
                    save_model(model, output_dir, 'QuesNet', '%d.%d' % (epoch, i))

                if train_params['log_steps'] > 0 and i % train_params['log_steps'] == 0:
                    print(f"{epoch}.{i}---loss: {loss.item()}")

            save_model(model, output_dir, 'QuesNet', '%d' % (epoch + 1))

        except KeyboardInterrupt:
            raise