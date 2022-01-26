from msilib.schema import Error
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..ModelZoo.DisenQNet.DisenQNet import DisenQNet, ConceptModel
from ..ModelZoo.DisenQNet.dataset import QuestionDataset
from ..Tokenizer import PureTextTokenizer
from ..SIF import Symbol, FORMULA_SYMBOL, FIGURE_SYMBOL, QUES_MARK_SYMBOL, TAG_SYMBOL, SEP_SYMBOL


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


class DisenTokenizer(object):
    def __init__(self, vocab_path=None, max_length=250,*args):
        super(DisenTokenizer, self).__init__(*args)
        self.num_token = "<num>"
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"

        self.max_length = max_length
        self.word2index = self.get_vocab(vocab_path)

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
      items = text_tokenizer(items)
      
      token_items = list()
      stop_words = set("\n\r\t .,;?\"\'。．，、；？“”‘’（）")
      for text in items:
          # text = key(data).strip().split(' ')
          print("1 ",text)
          text = [w for w in text if w != '' and w not in stop_words]
          print("2 ",text)
          if len(text) == 0:
              text = [self.unk_token]
          if len(text) > self.max_length:
              text = text[:self.max_length]
          text = [self.num_token if check_num(w) else w for w in text]
          print("3 ",text)
          token_items.append(text)
          
      return token_items


def train(items, output_dir, train_params=None):
  """
  Parameters
  ----------


  """
  # dataset
  train_path = os.path.join(args.dataset, "train_small.json")
  test_path = os.path.join(args.dataset, "test.json")
  wv_path = os.path.join(args.dataset, "wv.th")
  word_path = os.path.join(args.dataset, "vocab.list")
  concept_path = os.path.join(args.dataset, "concept.list")

  train_dataset = QuestionDataset(train_path, wv_path, word_path, concept_path, args.hidden, args.trim_min, args.max_len, "train", silent=False)
  test_dataset = QuestionDataset(test_path, wv_path, word_path, concept_path, args.hidden, args.trim_min, args.max_len, "test", silent=False)
  train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=train_dataset.collate_data)
  test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=test_dataset.collate_data)

  # model 
  vocab_size = train_dataset.vocab_size
  concept_size = train_dataset.concept_size
  wv = train_dataset.word2vec


  disen_q_net = DisenQNet(vocab_size, concept_size, args.hidden, args.dropout, args.pos_weight, args.cp, args.mi, args.dis, wv)
  pass
