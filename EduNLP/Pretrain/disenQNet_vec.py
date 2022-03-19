from msilib.schema import Error
import os
from re import A
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..ModelZoo.DisenQNet.DisenQNet import DisenQNet, ConceptModel
from ..ModelZoo.DisenQNet.dataset import QuestionDataset
from ..Tokenizer import PureTextTokenizer
from ..SIF import Symbol, FORMULA_SYMBOL, FIGURE_SYMBOL, QUES_MARK_SYMBOL, TAG_SYMBOL, SEP_SYMBOL
import json


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


class DisenQTokenizer(object):
    def __init__(self, vocab_path=None, conifg_dir=None, max_length=250,*args):
        super(DisenQTokenizer, self).__init__(*args)

        self.num_token = "<num>"
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        if conifg_dir is not None:
            self.load_tokenizer_config(conifg_dir)
        else:
            self.max_length = max_length
        
        self.word2index = self.get_vocab(vocab_path)

    def load_tokenizer_config(self, conifg_dir):
        with open(conifg_dir, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            self.max_length = model_config["max_len"]

    def __call__(self, items: (list, str), key=lambda x: x, padding=True, return_tensors=True, *args, **kwargs):
        print("test items : ",items)
        token_items = self.tokenize(items)
        print("test tokenize : ",token_items)
        # index
        ids = list()
        lengths = list()
        for item in token_items:
          indexs = [self.word2index.get(w, self.word2index[self.unk_token]) for w in item]
          ids.append(indexs)
          lengths.append(len(indexs))
        
        print("test ids : ",ids)
        print("test lengths : ",lengths)

        ret = {
          "input_ids": self.padding(ids) if padding else ids,
          "input_lens":lengths
        }
        print("ret",ret)
        return {key: torch.as_tensor(val) for key, val in ret.items()} if return_tensors else ret


    def tokenize(self, items: (list, str), key=lambda x: x, *args, **kwargs):
      if isinstance(items, str):
        return self._tokenize([items], key)
      elif isinstance(items, list):
        return self._tokenize(items, key)
      else:
         raise ValueError("items should be str or list!")

    def get_vocab(self, path):
      with open(path, "rt", encoding="utf-8") as file:
        items = file.read().strip().split('\n')
        item2index = {item:index for index, item in enumerate(items)}
      return item2index

    @property
    def vocab_size(self):
        return len(self.word2index)

    def padding(self, ids):
      max_len = max([len(i) for i in ids])
      padding_ids = [t + [ self.word2index[self.unk_token]] * (max_len - len(t) ) for t in ids]
      return padding_ids
      

    def _tokenize(self, items, key=lambda x: x):
      text_tokenizer = PureTextTokenizer()
      items = text_tokenizer(items, key=key)
      
      token_items = list()
      stop_words = set("\n\r\t .,;?\"\'。．，、；？“”‘’（）")
      for text in items:
          # text = key(data).strip().split(' ')
          print("1 ",text)
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


def load_data(path):
    if not os.path.exists(path):
        return None
    with open(path, "rt", encoding="utf-8") as file:
        all_data = list()
        for line in file:
            data = json.loads(line)
            all_data.append(data)
    return all_data


def train_disenQNet(train_items, output_dir, train_params=None, predata_path=None, test_items=None):
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
    if train_params is not None:
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

            "device": "cpu",
        }
        train_params = default_train_params.update(train_params)

    # wv_path = os.path.join(data_path, "wv.th")
    # word_path = os.path.join(data_path, "vocab.list")
    # concept_path = os.path.join(data_path, "concept.list")
    use_predata_path = True if predata_path is not None and os.path.exists(predata_path) else False
    wv_path = os.path.join(predata_path, "wv.th") if use_predata_path else None
    word_path = os.path.join(predata_path, "vocab.list") if use_predata_path else None
    concept_path = os.path.join(predata_path, "concept.list") if use_predata_path else None

    # train_items = load_data(os.path.join(data_path, "train.json")) # can replay by items
    # assert train_items is not None
    train_dataset = QuestionDataset(train_items, train_params.max_len, "train", silent=False,
                                    embed_dim=train_params.hidden, trim_min_count=train_params.trim_min, 
                                    wv_path=wv_path, word_path=word_path, concept_path=concept_path)
    train_dataloader = DataLoader(train_dataset, batch_size=train_params.batch, shuffle=True, collate_fn=train_dataset.collate_data)

    # test_items = load_data(os.path.join(data_path, "test.json"))
    if test_items is not None:
        test_dataset = QuestionDataset(test_items, train_params.max_len, "test", silent=False,
                                        embed_dim=train_params.hidden, trim_min_count=train_params.trim_min,
                                        wv_path=wv_path, word_path=word_path, concept_path=concept_path)
        test_dataloader = DataLoader(test_dataset, batch_size=train_params.batch, shuffle=False, collate_fn=test_dataset.collate_data)
    else:
        test_dataloader = None
    
    vocab_size = train_dataset.vocab_size
    concept_size = train_dataset.concept_size
    wv = train_dataset.word2vec

    # model
    disen_q_net = DisenQNet(vocab_size, concept_size, train_params.hidden, train_params.dropout, train_params.pos_weight, train_params.cp, train_params.mi, train_params.dis, wv)

    disen_q_net.train(train_dataloader, test_dataloader, train_params.device, train_params.epoch, train_params.lr, train_params.step, train_params.gamma, train_params.warm_up, train_params.adv, silent=False)
    disen_q_net_path = os.path.join(output_dir, "disen_q_net.th")
    disen_q_net.save(disen_q_net_path)
