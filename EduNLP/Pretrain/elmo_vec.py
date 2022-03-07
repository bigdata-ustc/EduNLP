import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as tud
import torch.optim as optim
import numpy as np
import json
import os
import time
from EduNLP.Pretrain import BertTokenizer
from EduNLP.SIF import Symbol, FORMULA_SYMBOL, FIGURE_SYMBOL, QUES_MARK_SYMBOL, TAG_SYMBOL, SEP_SYMBOL
from EduNLP.Tokenizer import PureTextTokenizer
from EduNLP.ModelZoo.rnn import ElmoLM

UNK_SYMBOL = '[UNK]'
PAD_SYMBOL = '[PAD]'


class ElmoTokenizer(object):
    """

    Examples
    --------
    >>> vocab=ElmoTokenizer()
    >>> items = ["有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"]
    >>> vocab.tokenize(items[0])
    ['公式', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[SEP]', 'z', '=', 'x', '+', '7', 'y', '最大值', '[MARK]']
    >>> len(vocab)
    18
    """

    def __init__(self, path=None):
        self.pure_tokenizer = PureTextTokenizer()
        self.t2id = {PAD_SYMBOL: 0, UNK_SYMBOL: 1, FORMULA_SYMBOL: 2, FIGURE_SYMBOL: 3,
                     QUES_MARK_SYMBOL: 4, TAG_SYMBOL: 5, SEP_SYMBOL: 6}
        if path is None:
            pass
        else:
            self.load_vocab(path)

    def __call__(self, item, *args, **kwargs):
        return self.to_index(self.tokenize(item))

    def __len__(self):
        return len(self.t2id)

    def tokenize(self, item: str):
        tokens = next(self.pure_tokenizer([item]))
        for token in tokens:
            self.append(token)
        return tokens

    def to_index(self, item: list, max_length=128, pad_to_max_length=False):
        ret = [self.t2id[UNK_SYMBOL] if token not in self.t2id else self.t2id[token] for token in item]
        if pad_to_max_length:
            if len(ret) < max_length:
                ret = ret + (max_length - len(ret)) * [self.t2id[PAD_SYMBOL]]
            else:
                ret = ret[0:max_length - 1]
        return ret

    def append(self, item):
        if item in self.t2id:
            pass
        else:
            self.t2id[item] = len(self.t2id)

    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump(self.t2id, f)
        return path

    def load_vocab(self, path):
        with open(path, 'r') as f:
            self.t2id = json.load(f)
        return path


class ElmoDataset(tud.Dataset):
    def __init__(self, texts: list, tokenizer: ElmoTokenizer, max_length=128):
        super(ElmoDataset, self).__init__()
        self.tokenizer = tokenizer
        self.texts = [text if len(text) < max_length else text[0:max_length - 1] for text in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        sample = {
            'length': len(text),
            'idx': self.tokenizer.to_index(text, pad_to_max_length=True)
        }
        return sample


def elmo_collate_fn(batch_data):
    # batch_data = [torch.tensor(t).cuda() for t in batch_data]
    # batch_data = torch.nn.utils.rnn.pad_sequence(batch_data)
    pred_mask = []
    idx_mask = []
    max_len = max([data['length'] for data in batch_data])
    for data in batch_data:
        pred_mask.append([True] * data['length'] + [False] * (max_len - data['length']))
    for data in batch_data:
        idx_mask.append([True] * data['length'] + [False] * (len(data['idx']) - data['length']))
    ret_batch = {
        'pred_mask': torch.tensor(pred_mask),
        'idx_mask': torch.tensor(idx_mask),
        'length': torch.tensor([data['length'] for data in batch_data]),
        'idx': torch.tensor([data['idx'] for data in batch_data])
    }
    return ret_batch


def train_elmo(texts, output_dir, pretrained_dir: str = None, emb_dim=512, hid_dim=1024, batch_size=2,
               epochs=3, lr: float = 5e-4):
    """
    Parameters
    ----------
    texts: list, required
        The training corpus of shape (text_num, token_num), a text must be tokenized into tokens
    output_dir: str, required
        The directory to save trained model files
    pretrained_dir: str, optional
        The pretrained model files' directory
    emb_dim: int, optional, default=512
        The embedding dim
    hid_dim: int, optional, default=1024
        The hidden dim
    batch_size: int, optional, default=2
        The training batch size
    epochs: int, optional, default=3
        The training epochs
    lr: float, optional, default=5e-4
        The learning rate

    Returns
    -------
    output_dir: str
        The directory that trained model files are saved
    """
    tokenizer = ElmoTokenizer()
    if pretrained_dir:
        tokenizer.load_vocab(os.path.join(pretrained_dir, 'vocab.json'))
    else:
        for text in texts:
            for token in text:
                tokenizer.append(token)
    train_dataset = ElmoDataset(texts, tokenizer)
    if pretrained_dir:
        model = torch.load(os.path.join(pretrained_dir, 'weight.pt'))
    else:
        model = ElmoLM(vocab_size=len(tokenizer), embedding_dim=emb_dim, hidden_size=hid_dim, )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.LM_layer.rnn.flatten_parameters()
    model.to(device)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model.train()
    global_step = 0
    adam = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCELoss()
    loss_func.to(device)
    dataloader = tud.DataLoader(train_dataset, collate_fn=elmo_collate_fn, batch_size=batch_size, shuffle=True)
    begin = time.time()
    for epoch in range(epochs):
        for step, sample in enumerate(dataloader):
            pred_mask = sample['pred_mask'].to(device)
            idx_mask = sample['idx_mask'].to(device)
            idx = sample['idx'].to(device)
            length = sample['length'].to(device)
            try:
                y = F.one_hot(idx, num_classes=len(tokenizer)).to(device)
                pred_forward, pred_backward, _, _ = model.forward(idx, length, device)
                pred_forward = pred_forward[pred_mask]
                pred_backward = pred_backward[torch.flip(pred_mask, [1])]
                y_rev = torch.flip(y, [1])[torch.flip(idx_mask, [1])]
                y = y[idx_mask]
                forward_loss = loss_func(pred_forward[:, :-1].double(), y[:, 1:].double())
                backward_loss = loss_func(pred_backward[:, :-1].double(), y_rev[:, 1:].double())
                forward_loss.backward(retain_graph=True)
                backward_loss.backward(retain_graph=True)
                adam.step()
                adam.zero_grad()
                global_step += 1
                if global_step % 10 == 0:
                    print("[Global step %d, epoch %d, batch %d] Loss: %.10f" % (
                        global_step, epoch, step, forward_loss + backward_loss))
            except RuntimeError as e:
                print("RuntimeError:", e)
                print("[DEBUG]Sample idx:", idx)
    end = time.time()
    print("Train time: ", (end - begin))
    model.cpu()
    config = {
        'emb_dim': emb_dim,
        'hid_dim': hid_dim,
        'batch_first': True
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f)
    torch.save(model.state_dict(), os.path.join(output_dir, 'weight.pt'))
    tokenizer.save_vocab(os.path.join(output_dir, 'vocab.json'))
    return output_dir
