from .pretrian_utils import PretrainedEduTokenizer
from ..SIF.segment.segment import FigureSegment
from ..ModelZoo.quesnet import QuesNetForPreTraining, AE
from EduNLP import logger

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_grayscale
from torchvision.transforms.functional import to_tensor
from gensim.models import Word2Vec
import torch
import re
import warnings
import queue
import random
import math
import threading
import logging
import signal
import os
import json
import copy
import linecache
import numpy as np
from PIL import Image
from typing import List, Union, Optional
from collections import namedtuple
from functools import partial
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def save_list(item2index, path):
    item2index = sorted(item2index.items(), key=lambda kv: kv[1])
    items = [item for item, _ in item2index]
    with open(path, "wt", encoding="utf-8") as file:
        file.write('\n'.join(items))
    return


def clip(v, low, high):
    return max(low, min(v, high))


# Basic unit of Dataset
Question = namedtuple('Question',
                      ['id', 'content', 'answer', 'false_options', 'labels'])


class QuesNetTokenizer(PretrainedEduTokenizer):
    """
    Examples
    --------
    >>> tokenizer = QuesNetTokenizer(meta=['knowledge'])
    >>> test_items = [{"ques_content": "$\\triangle A B C$ 的内角为 $A, \\quad B, $\\FigureID{test_id}$",
    ... "knowledge": "['*', '-', '/']"}, {"ques_content": "$\\triangle A B C$ 的内角为 $A, \\quad B",
    ... "knowledge": "['*', '-', '/']"}]
    >>> tokenizer.set_text_vocab(test_items,
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
                 meta: List[str] = None, img_token='<img>', unk_token="<unk>", pad_token="<pad>", **kwargs):
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
        if add_specials is None or isinstance(add_specials, bool):
            add_specials = [img_token]
        else:
            add_specials = [img_token] + add_specials

        self.tokenization_params = {
            "formula_params": {
                "method": "linear",
                "skip_figure_formula": True
            }
        }
        self.tokenization_params.update(kwargs)
        super().__init__(vocab_path=vocab_path, max_length=max_length, tokenize_method=tokenize_method,
                         add_specials=add_specials, unk_token=unk_token, pad_token=pad_token, symbol=symbol, **self.tokenization_params)
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
                "self", "__class__", 'img_dir', 'vocab_path', 'meta_vocab_dir']
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
            if isinstance(w, FigureSegment) and isinstance(item, dict) and 'ques_figure_ids' in item.keys():
                # image

                try:
                    fig_id = f"{w.src[10:-1]}"
                    fig_index = item['ques_figure_ids'].index(fig_id)

                    if self.img_dir is not None:
                        fig_src = os.path.join(self.img_dir, fig_id)
                        if '.png' in item['ques_figure_paths'][fig_index]:
                            fig_src += '.png'
                        elif '.jpg' in item['ques_figure_paths'][fig_index]:
                            fig_src += '.jpg'
                    else:
                        fig_src = item['ques_figure_paths'][fig_index]
                    
                    im = Image.open(fig_src)
                    im = im.resize((56, 56))
                    token_idx.append(to_grayscale(im))
                    # logger.info(f"Open figure {fig_src}")
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
                warnings.warn(f"load_meta_vocab Error!!!, SET stoi[{m}] = None")
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

    def set_text_vocab(self, items: list, key=lambda x: x, lower: bool = False,
                  trim_min_count: int = 1, do_tokenize: bool = True, symbol=None, silent=True):
        """
        Parameters
        -----------
        items: list
            can be the list of str, or list of dict
        key: function
            determine how to get the text of each item
        trim_min_count : int, optional
            the lower bound number for adding a word into vocabulary, by default 1
        silent
        """
        token_items = self.tokenize(items, key=key, symbol=symbol) if do_tokenize else [key(item) for item in items]
        self.vocab.set_vocab(corpus_items=token_items, trim_min_count=trim_min_count, lower=lower, silent=silent)
        self.stoi['word'] = self.vocab.token_to_idx
        self.itos['word'] = self.vocab.idx_to_token
        return token_items

    def set_vocab(self, items: list, key=lambda x: x, lower: bool = False, meta: List[str] = None,
                  trim_min_count: int = 1, do_tokenize: bool = True, symbol=None, silent=True,):
        token_items = self.set_text_vocab(items, key=key, lower=lower, trim_min_count=trim_min_count, do_tokenize=do_tokenize,
                            symbol=symbol, silent=silent)
        self.set_meta_vocab(items, meta=meta, silent=silent)
        return token_items
        
    @classmethod
    def from_pretrained(cls, tokenizer_config_dir, img_dir=None, **kwargs):
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
            tokenizer_config.update(kwargs)
            return cls(
                vocab_path=pretrained_vocab_path,
                meta_vocab_dir=tokenizer_config_dir,
                img_dir=img_dir,
                **tokenizer_config)

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


class QuesnetDataset(Dataset):
    '''
        Quesnet-specific datasets
    '''
    def __init__(
        self,
        items=None,
        filename: str=None,
        tokenizer: str=None,
        content_key=lambda x: x['ques_content'],
        answer_key=lambda x: x['ques_answer'],
        option_key=lambda x: x['ques_options'],
        pipeline=None,
        skip=0
    ):
        self.filename = filename
        self.skip = skip
        self.content_key = content_key
        self.answer_key = answer_key
        self.option_key = option_key
        self.pipeline = pipeline
        if items is None and filename is not None:
            self.load_data_lines()
        else:
            self.lines = items
        
        self.tokenizer = tokenizer
        self.meta = tokenizer.meta

    def load_data_lines(self):
        '''Read data by row from a JSON file

        Important: the data file is loaded during initialization.
        '''

        # TODO: All data is read into memory without chunking.
        #       This may lead to low efficiency.
        data_dir = self.filename
        skip = self.skip        # Read from Line skip + 1.
        self.lines = []
        self.length = 0

        with open(data_dir, "r", encoding="utf-8") as f:
            row = 0
            while True:
                row += 1
                line = f.readline()
                if row <= skip:
                    continue
                if not line:
                    break
                self.lines.append(json.loads(line.strip()))

            self.length = row - skip - 1
        assert self.length > 0, f'{data_dir} is empty. Or file length is less than skip length.'

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        if isinstance(index, int):
            line = self.lines[index]

            qid = line['ques_id']
            token = self.tokenizer(line, key=self.content_key, meta=self.meta)
            content = token['seq_idx']
            meta = token['meta_idx']
            raw_answer = self.answer_key(line)
            # Fliter $A$
            raw_answer = raw_answer.strip("$") if isinstance(raw_answer, str) else raw_answer
            if raw_answer.isalpha() and len(raw_answer) == 1 \
                    and ord(raw_answer) < 128 and len(self.option_key(line)) > 0:
                answer_idx = ord(raw_answer.upper()) - ord('A')
                options = self.option_key(line)
                # Only Suport A,B,C,D FORMAT
                if len(options) == 4 and answer_idx <=3 and  answer_idx>=0:
                    answer = self.tokenizer(options.pop(answer_idx), meta=self.meta)['seq_idx']
                    false_options = [(self.tokenizer(option, meta=self.meta))['seq_idx'] for option in options]
                else:
                    # logger.info(f"[warn: Not GOOD OPTIONS and ANSWER] answer={raw_answer}, options={options}")
                    answer = self.tokenizer(raw_answer, meta=self.meta)['seq_idx']
                    false_options = [[0], [0], [0]]  
            else:
                answer = self.tokenizer(raw_answer, meta=self.meta)['seq_idx']
                false_options = [[0], [0], [0]]

            qs = Question(
                id=qid,
                content=content,
                answer=answer,
                false_options=false_options,
                labels=meta
            )
            # if callable(self.pipeline):
            #     qs = self.pipeline(qs)

            return qs

        elif isinstance(index, slice):
            results = []
            for i in range(*index.indices(len(self))):
                results.append(self[i])
            return results

        else:
            raise TypeError('Invalid argument type. Index type should be int or slice.')


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


class PrefetchIter:
    """Iterator on data and labels, with states for save and restore."""

    def __init__(self, data, *label, length=None, batch_size=1, shuffle=True):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=8)
        self.length = length if length is not None else len(data)
        self.pipeline = data.pipeline

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

                # if callable(self.pipeline):
                #     data_batch = self.pipeline(data_batch)

                label_batch = [label[index * bs:(index + 1) * bs]
                               for label in self.label]
                if label_batch:
                    self.queue.put([data_batch] + label_batch)
                else:
                    self.queue.put(data_batch)
            except Exception as e:
                self.queue.put(e)
                return


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


def pretrain_embedding_layer(dataset: EmbeddingDataset, ae: AE, lr: float = 1e-3, log_step: int = 10, epochs: int = 3,
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


def optimizer(*models, **kwargs):
    _cur_optim = [
        m.optim_cls(m.parameters(), **kwargs)
        if hasattr(m, 'optim_cls')
        else torch.optim.Adam(m.parameters(), **kwargs) for m in models
    ]
    if len(_cur_optim) == 1:
        return _cur_optim[0]
    else:
        return _cur_optim


def pretrain_quesnet(
    train_items,
    output_dir,
    pretrain_dir=None,
    img_dir=None,
    save_embs=True,
    load_embs=False,
    train_params=None,
    w2v_params=None, 
    pretrained_wv=None,
    data_params=None,
    model_params=None,
    tokenizer_params=None,
    dataset=None,
    tokenizer=None
):
    """ pretrain quesnet

    Parameters
    ----------
    train_items : str
        questions
    output_dir : str
        output path·
    tokenizer : QuesNetTokenizer
        quesnet tokenizer
    save_embs : bool, optional
        whether to save pretrained word/image/meta embeddings seperately
    load_embs : bool, optional
        whether to load pretrained word/image/meta embeddings seperately
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
    >>> tokenizer.set_text_vocab(items, key=lambda x: x['ques_content'], trim_min_count=1, silent=True)
    >>> pretrain_quesnet('./data/standard_luna_data.json', './testQuesNet', tokenizer) # doctest: +SKIP
    """
    tokenizer_params = tokenizer_params if tokenizer_params else {}
    data_params = data_params if data_params is not None else {}
    model_params = model_params if model_params is not None else {}
    train_params = train_params if train_params is not None else {}
    w2v_params = w2v_params if w2v_params is not None else {}
    pretrain_dir = "" if pretrain_dir is None else pretrain_dir
    
    data_formation = {
        "ques_content": "ques_content",
        "know_name": "know_name",
        "ques_answer": "ques_answer",
        "ques_options": "ques_options",
        "meta": ['know_name']
    }
    data_formation.update(data_params.get("data_formation", {}))

    os.makedirs(output_dir, exist_ok=True)
    tensorboard_dir = f'{output_dir}/tensorboard'
    os.makedirs(tensorboard_dir, exist_ok=True)
    tensorboard_writer = SummaryWriter(tensorboard_dir)

    default_w2v_params = {
        "epochs": 1,
        "min_count": 2,
        "workers": 2,
    }
    if w2v_params is not None:
        default_w2v_params.update(w2v_params)
    w2v_params = default_w2v_params

    work_train_params = {
        # train params
        "n_epochs": 1,
        "batch_size": 8,
        "lr": 1e-3,
        'save_every_steps': 0,
        'save_every_epochs': 1,
        'log_steps': 10,
        'device': 'cpu',
        'max_steps': 0,
        # model params
        'emb_size': 256,
        'feat_size': 256,
    }
    work_train_params.update(train_params)


    work_model_params = {
        'emb_size': 256,
        'feat_size': 256,
    }
    work_model_params.update(model_params)

    device = torch.device(work_train_params['device'])
    token_items = None
    if tokenizer is None:
        tokenizer = QuesNetTokenizer(
            meta=data_formation["meta"],
            img_dir=img_dir,
            **tokenizer_params,
        )
        # token_items = 
        tokenizer.set_vocab(items=train_items,
                            key=data_formation["ques_content"],
                            trim_min_count=data_params.get("trim_min_count", 2),
                            silent=False,
                            meta=data_formation["meta"]
                            )
    elif img_dir is not None:
        tokenizer.set_img_dir(img_dir)
    
    # if token_items is None:
    #     # Note that: must set symbol = 'g...' to tokenize Figure as []
    #     symbol = tokenizer.config["symbol"]
    #     if not re.findall(r"g", symbol):
    #         symbol = "g" + symbol
    #     token_items = tokenizer.tokenize(train_items,
    #                                      key=lambda x: x[data_formation["ques_content"]],
    #                                      symbol=symbol)

    tokenizer.save_pretrained(output_dir)

    if dataset is None:
        dataset = QuesnetDataset(items=train_items,
                                 tokenizer=tokenizer,
                                 content_key=lambda x : x[data_formation["ques_content"]],
                                 option_key=lambda x : x[data_formation["ques_options"]],
                                 answer_key=lambda x : x[data_formation["ques_answer"]],
                                 )
    logger.info(f"dataset: {len(dataset.lines)}")
    logger.info(f"dataset[0]: {dataset[0]}")

    model = QuesNetForPreTraining(_stoi=tokenizer.stoi, feat_size=work_model_params['feat_size'],
                                  emb_size=work_model_params['emb_size']).to(device)

    emb_dict_rev = tokenizer.itos['word']
    emb_size = work_model_params['emb_size']
    meta_size = model.quesnet.meta_size
    w2v_corpus = []
    img_corpus = []
    meta_corpus = []

    for _, qs in enumerate(tqdm(dataset)):
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

    logger.info("train start!")
    # train word2vec for text embedding
    w2v_auto_model_path=os.path.join(pretrain_dir, 'w2v_embs.npy')
    if pretrain_dir and load_embs and os.path.exists(w2v_auto_model_path):
        model.quesnet.load_emb(np.load(w2v_auto_model_path))
    else:
        # gensim_w2v = Word2Vec(
        #     sentences=[[item] for item in emb_dict.keys()],
        #     vector_size=emb_size,
        #     min_count=w2v_params["min_count"],
        #     workers=w2v_params["workers"]
        # )
        # gensim_w2v.init_weights()
        # gensim_w2v.train(corpus_iterable=w2v_corpus, total_examples=len(w2v_corpus), epochs=w2v_params['epochs'])
        # w2v_emb = gensim_w2v.syn1neg
        # emb_weights = []
        # for key, item in emb_dict.items():
        #     w2v_index = gensim_w2v.wv.key_to_index[key]
        #     emb_weights.append(w2v_emb[w2v_index])

        words = tokenizer.vocab.tokens
        # unk_token = tokenizer.vocab.unk_token
        # corpus = list()
        # word_set = set(words)
        # for text in token_items:
        #     text = [w if w in word_set else unk_token for w in text]
        #     corpus.append(text)
        
        logger.info(f"w2v_corpus: {len(w2v_corpus)}")
        # logger.info(f"token_items: {len(token_items)}")
        # logger.info(f"corpus: {len(corpus)}")
        if pretrained_wv is None:
            wv = Word2Vec(w2v_corpus, vector_size=emb_size, **w2v_params).wv
            logger.info("train new pretrained_wv !")
        else:
            wv = pretrained_wv
            logger.info("load edunlp pretrained_wv !")
        # 按照 vocab 中的词序 来保存
        emb_weights = [wv[w] if w in wv.key_to_index else np.random.rand(emb_size) for w in words]
        if save_embs:
            np.save(w2v_auto_model_path, emb_weights)
        model.quesnet.load_emb(np.array(emb_weights))
    logger.info('quesnet Word Embedding loaded')

    # train auto-encoder loss for image embedding
    img_auto_model_path=os.path.join(pretrain_dir, 'trained_ie.pt')
    if pretrain_dir and load_embs and os.path.exists(img_auto_model_path):
        model.quesnet.load_img(torch.load(img_auto_model_path))
    else:
        logger.info(f"img_corpus: {len(img_corpus)}")
        img_dataset = EmbeddingDataset(data=img_corpus, data_type='image')
        trained_ie = pretrain_embedding_layer(
            dataset=img_dataset,
            ae=model.quesnet.ie,
            lr=work_train_params['lr'],
            log_step=work_train_params['log_steps'],
            batch_size=work_train_params['batch_size'],
            epochs=work_train_params['n_epochs'],
            device=device
        )
        if save_embs:
            torch.save(trained_ie.state_dict(), img_auto_model_path)
        model.quesnet.load_img(trained_ie.state_dict())
    logger.info('quesnet Image Embedding loaded')

    # train auto-encoder loss for meta embedding
    meta_auto_model_path = os.path.join(pretrain_dir, 'trained_me.pt')
    if pretrain_dir and load_embs and os.path.exists(meta_auto_model_path):
        model.quesnet.load_meta(torch.load(meta_auto_model_path))
    else:
        logger.info(f"meta_corpus: {len(meta_corpus)}")
        meta_dateset = EmbeddingDataset(data=meta_corpus, data_type='meta')
        trained_me = pretrain_embedding_layer(
            dataset=meta_dateset,
            ae=model.quesnet.me,
            lr=work_train_params['lr'],
            log_step=work_train_params['log_steps'],
            batch_size=work_train_params['batch_size'],
            epochs=work_train_params['n_epochs'],
            device=device
        )
        if save_embs:
            torch.save(trained_me.state_dict(), meta_auto_model_path)
        model.quesnet.load_meta(trained_me.state_dict())
    logger.info('quesnet Meta Embedding loaded')

    logger.info("quesnet Word, Image and Meta Embeddings training is done")
    # DONE for datasets

    # debug
    # device = torch.device("cpu")
    pretrained_model_path = os.path.join(pretrain_dir, 'pytorch_model.bin')
    if pretrain_dir and load_embs and os.path.exists(pretrained_model_path):
        # For continuing training
        model = QuesNetForPreTraining.from_pretrained(pretrain_dir)
        logger.info(f"Load QuesNetForPreTraining from checkpoint: {pretrained_model_path}")

    # HLM and DOO training
    dataset.pipeline = partial(model.quesnet.make_batch, device=device, pretrain=True)
    model.train()
    optim = optimizer(model, lr=work_train_params['lr'])
    n_batches = 0
    n_steps = 0
    for epoch in range(0, work_train_params['n_epochs']):
        train_iter = PrefetchIter(dataset, batch_size=work_train_params['batch_size'])
        bar = enumerate(tqdm(train_iter, initial=train_iter.pos),
                        train_iter.pos)
        for i, batch in critical(bar):
            batch = model.quesnet.make_batch(batch, device=device, pretrain=True)
            loss = model(batch).loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            tensorboard_writer.add_scalar("train_loss", loss.item(), n_steps)
            n_steps += 1

            if work_train_params['save_every_steps'] > 0 and n_steps % work_train_params['save_every_steps'] == 0:
                # model.save(os.path.join(output_dir, f'QuesNet_{epoch}.{i}'))
                model.save_pretrained(f"{output_dir}/checkpoint-S{n_steps}")

            if work_train_params['log_steps'] > 0 and i % work_train_params['log_steps'] == 0:
                logger.info(f"{epoch}.{i}---loss: {loss.item()}")

            if work_train_params['max_steps'] > 0 and n_steps % work_train_params['max_steps'] == 0:
                break
        n_batches += 1
         if work_train_params['save_every_epochs'] > 0 and n_batches % work_train_params['save_every_epochs'] == 0:
            # model.save(os.path.join(output_dir, f'QuesNet_{epoch}.{i}'))
            model.save_pretrained(f"{output_dir}/checkpoint-E{n_batches}") 
            
        # model.save(os.path.join(output_dir, f'QuesNet_{epoch}'))

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
