import os
import numpy as np
import torch
from copy import deepcopy
from transformers import TrainingArguments, Trainer
import warnings
from .pretrian_utils import EduDataset
from typing import Dict, List, Tuple, Union, Any
from gensim.models import Word2Vec
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from dataclasses import dataclass, field
from ..SIF import EDU_SPYMBOLS
from ..ModelZoo.disenqnet.disenqnet import DisenQNetForPreTraining
from ..ModelZoo.utils import load_items, pad_sequence
from .pretrian_utils import PretrainedEduTokenizer


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
    ...     key=lambda x: x["content"], trim_min_count=1)
    [['甲', '数', '除以', '乙', '数', '商', '[NUM]', '甲', '数', '增加', '[NUM]', '甲', '数', '乙', '倍', '甲', '数']]
    >>> token_items = [tokenizer(i, key=lambda x: x["content"]) for i in test_items]
    >>> print(token_items[0].keys())
    dict_keys(['seq_idx', 'seq_len'])
    """

    def __init__(self, vocab_path=None, max_length=250, tokenize_method="pure_text",
                 add_specials: list = None, num_token="[NUM]", **kwargs):
        """
        Parameters
        ----------
        vocab_path: str
            default is None
        max_length: int
            default is 250, used to clip the sentence out of length
        tokenize_method: str
            default: "pure_text"
            when text is already seperated by space, use "space"
            when text is raw string format, use Tokenizer defined in get_tokenizer(), such as "pure_text" and "text"
        num_token: str
        """
        if add_specials is None:
            add_specials = [num_token]
        else:
            add_specials = [num_token] + add_specials
        super().__init__(vocab_path=vocab_path, max_length=max_length,
                         tokenize_method=tokenize_method, add_specials=add_specials, **kwargs)
        self.num_token = num_token
        self.config = {k: v for k, v in locals().items() if k not in ["self", "__class__", "vocab_path"]}

    def _tokenize(self, item: Tuple[str, dict], key=lambda x: x):
        token_item = self.text_tokenizer._tokenize(item, key=key)
        if len(token_item) == 0:
            token_item = [self.vocab.unk_token]
        if len(token_item) > self.max_length:
            token_item = token_item[:self.max_length]
        token_item = [self.num_token if check_num(w) else w for w in token_item]
        return token_item


def preprocess_dataset(pretrained_dir, disen_tokenizer, items, data_formation, trim_min_count=None, embed_dim=None,
                       w2v_params=None, silent=False):
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
        token_items = disen_tokenizer.set_vocab(items, key=lambda x: x[data_formation["ques_content"]],
                                                trim_min_count=trim_min_count)
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
            print(data)
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


class DisenQDataset(EduDataset):
    def __init__(self, items: List[Dict], tokenizer: DisenQTokenizer, data_formation: Dict,
                 mode="train", concept_to_idx=None, **kwargs):
        """
        Parameters
        ----------
        texts: list
        tokenizer: DisenQTokenizer
        data_formation: dict
        max_length: int, optional, default=128
        """
        # super(DisenQDataset, self).__init__(tokenizer=tokenizer, **kwargs)
        self.tokenizer = tokenizer
        self.concept_to_idx = concept_to_idx
        self.mode = mode
        self.items = items
        self.max_length = tokenizer.max_length
        self.data_formation = data_formation

    def __len__(self):
        return len(self.items)

    def _list_to_onehot(self, item_list, item2index):
        onehot = np.zeros(len(item2index)).astype(np.int64)
        for c in item_list:
            onehot[item2index[c]] = 1
        return onehot

    def __getitem__(self, index):
        item = self.items[index]
        ret = self.tokenizer(item, padding=False, key=lambda x: x[self.data_formation["ques_content"]],
                             return_tensors=False, return_text=False)
        if self.mode in ["train", "val"]:
            ret['concept'] = self._list_to_onehot(item[self.data_formation["knowledge"]], self.concept_to_idx)
        return ret

    def collate_fn(self, batch_data):
        pad_idx = self.tokenizer.vocab.pad_idx
        first = batch_data[0]
        batch = {
            k: [item[k] for item in batch_data] for k in first.keys()
        }
        batch["seq_idx"] = pad_sequence(batch["seq_idx"], pad_val=pad_idx)
        batch = {key: torch.as_tensor(val) for key, val in batch.items()}
        return batch


class DisenQTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = Adam(self.model.model_params, lr=self.args.learning_rate)
        self.lr_scheduler = StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        self.adv_optimizer = Adam(self.model.adv_params, lr=self.args.learning_rate)
        self.adv_scheduler = StepLR(self.adv_optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        warming_up = self.state.epoch <= model.params["warmup"]
        print(self.state)
        if not warming_up:
            # train disc
            outputs = model(**inputs)
            # stop gradient propagation to encoder
            k_hidden = outputs.k_hidden.detach()
            i_hidden = outputs.i_hidden.detach()
            # max dis_loss
            dis_loss = - model.disen_estimator(k_hidden, i_hidden)
            dis_loss = model.params["n_adversarial"] * model.params["w_dis"] * dis_loss
            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                dis_loss = dis_loss / self.args.gradient_accumulation_steps
            dis_loss.backward()
            step = self.state.global_step % (self.state.max_steps / self.state.num_train_epochs)
            if (step + 1) % self.args.gradient_accumulation_steps or \
                    (step + 1) == self.state.max_steps / self.state.num_train_epochs:
                self.adv_optimizer.step()
                self.adv_optimizer.zero_grad()
            # Lipschitz constrain for Disc of WGAN
            model.disen_estimator.spectral_norm()
        model.warming_up = warming_up

        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        # TODO: Not sure. if warming_up, the scheduler should not step?
        if not warming_up:
            self.adv_scheduler.step()

        return loss.detach()


@dataclass
class DisenQTrainingArguments(TrainingArguments):
    step_size: int = field(default=False, metadata={"help": "step_size"})
    trim_min: int = field(default=False, metadata={"help": "trim"})
    hidden_size: int = field(default=False, metadata={"help": "hidden_size"})
    gamma: float = field(default=False, metadata={"help": "gamma"})


DEFAULT_TRAIN_PARAMS = {
    # default
    "output_dir": None,
    "overwrite_output_dir": True,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 32,
    # evaluation_strategy: "steps",
    # eval_steps:200,
    "save_steps": 1000,
    "save_total_limit": 2,
    # "load_best_model_at_end": False,
    # metric_for_best_model: "loss",
    # greater_is_better: False,
    "logging_dir": None,
    "logging_steps": 5,
    "gradient_accumulation_steps": 1,
    "learning_rate": 5e-4,
    # disable_tqdm: True,
    # no_cuda: True,
    'step_size': 20,
    'gamma': 0.5,
    "trim_min": 1,
    "hidden_size": 300
}


def train_disenqnet(train_items: List[dict], output_dir: str, pretrained_dir: str = None,
                    eval_items=None, tokenizer_params=None, data_params=None, model_params=None,
                    train_params=None, w2v_params=None):
    """
    Parameters
    ----------
    train_items : List[dict]
        _description_
    output_dir : str
        _description_
    pretrained_dir : str, optional
        _description_, by default None
    tokenizer_params : _type_, optional
        _description_, by default None
    data_params : _type_, optional
        _description_, by default None
    model_params : _type_, optional
        _description_, by default None
    train_params : _type_, optional
        _description_, by default None
    """
    tokenizer_params = tokenizer_params if tokenizer_params else {}
    data_params = data_params if data_params is not None else {}
    model_params = model_params if model_params is not None else {}
    train_params = train_params if train_params is not None else {}
    w2v_params = w2v_params if w2v_params is not None else {}
    default_data_formation = {
        "ques_content": "ques_content",
        "knowledge": "knowledge"
    }
    data_formation = data_params.get("data_formation", None)
    if data_formation is not None:
        default_data_formation.update(data_formation)
    data_formation = default_data_formation

    # tokenizer configuration
    if pretrained_dir is not None and os.path.exists(pretrained_dir):
        tokenizer = DisenQTokenizer.from_pretrained(pretrained_dir, **tokenizer_params)
    else:
        work_tokenizer_params = {
            "add_specials": None,
            "tokenize_method": "pure_text",
        }
        work_tokenizer_params.update(tokenizer_params if tokenizer_params else {})
        tokenizer = DisenQTokenizer(**work_tokenizer_params)
        corpus_items = train_items
        tokenizer.set_vocab(corpus_items,
                            key=lambda x: x[data_formation['ques_content']])

    # training Configuration
    work_train_params = deepcopy(DEFAULT_TRAIN_PARAMS)
    work_train_params["output_dir"] = output_dir
    if train_params is not None:
        work_train_params.update(train_params if train_params else {})
    if model_params:
        if 'hidden_size' in model_params:
            work_train_params['hidden_size'] = model_params['hidden_size']

    # dataset configuration
    items = train_items + ([] if eval_items is None else eval_items)
    if pretrained_dir:
        tokenizer, concept_to_idx, word2vec = preprocess_dataset(pretrained_dir, tokenizer, items,
                                                                 data_formation,
                                                                 trim_min_count=work_train_params["trim_min"],
                                                                 embed_dim=work_train_params["hidden_size"],
                                                                 w2v_params=w2v_params, silent=False)
    else:
        tokenizer, concept_to_idx, word2vec = preprocess_dataset(output_dir, tokenizer, items,
                                                                 data_formation,
                                                                 trim_min_count=work_train_params["trim_min"],
                                                                 embed_dim=work_train_params["hidden_size"],
                                                                 w2v_params=w2v_params, silent=False)
    train_dataset = DisenQDataset(train_items, tokenizer, data_formation,
                                  mode="train", concept_to_idx=concept_to_idx)
    if eval_items:
        eval_dataset = DisenQDataset(eval_items, tokenizer, data_formation,
                                     mode="test", concept_to_idx=concept_to_idx)
    else:
        eval_dataset = None

    # model configuration
    if pretrained_dir:
        model = DisenQNetForPreTraining.from_pretrained(pretrained_dir, **model_params)
    else:
        work_model_params = {
            "vocab_size": len(tokenizer),
            "hidden_size": 300,
            'concept_size': len(concept_to_idx),
            'dropout_rate': 0.2,
            'pos_weight': 1,
            'w_cp': 1.5,
            'w_mi': 1.0,
            'w_dis': 2.0,
            'warmup': 1,
            'n_adversarial': 10,
            'gamma': 0.5,
            'step_size': 20,
            'wv': word2vec
        }
        work_model_params.update(model_params if model_params else {})
        model = DisenQNetForPreTraining(**work_model_params)

    # Train
    work_args = DisenQTrainingArguments(**work_train_params)
    trainer = DisenQTrainer(
        model=model,
        args=work_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_dataset.collate_fn,
    )
    trainer.train()
    # trainer.model.save_pretrained(output_dir)
    assert isinstance(trainer.model, DisenQNetForPreTraining)
    trainer.save_model(output_dir)
    trainer.model.save_config(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
