import torch
from torch.nn.utils.rnn import pack_padded_sequence


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


class SeqBatch:
    def __init__(self, seqs, dtype=None, device=None):
        self.dtype = dtype
        self.device = device
        self.seqs = seqs
        self.lens = [len(x) for x in seqs]

        self.ind = argsort(self.lens)[::-1]
        self.inv = argsort(self.ind)
        self.lens.sort(reverse=True)
        self._prefix = [0]
        self._index = {}
        c = 0
        for i in range(self.lens[0]):
            for j in range(len(self.lens)):
                if self.lens[j] <= i:
                    break
                self._index[i, j] = c
                c += 1

    def packed(self):
        ind = torch.tensor(self.ind, dtype=torch.long, device=self.device)
        padded = self.padded()[0].index_select(1, ind)
        return pack_padded_sequence(padded, torch.tensor(self.lens))

    def padded(self, max_len=None, batch_first=False):
        seqs = [torch.tensor(s, dtype=self.dtype, device=self.device)
                if not isinstance(s, torch.Tensor) else s
                for s in self.seqs]
        if max_len is None:
            max_len = self.lens[0]
        seqs = [s[:max_len] for s in seqs]
        mask = [[1] * len(s) + [0] * (max_len - len(s)) for s in seqs]

        trailing_dims = seqs[0].size()[1:]
        if batch_first:
            out_dims = (len(seqs), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(seqs)) + trailing_dims

        padded = seqs[0].new(*out_dims).fill_(0)
        for i, tensor in enumerate(seqs):
            length = tensor.size(0)
            # use index notation to prevent duplicate references to the tensor
            if batch_first:
                padded[i, :length, ...] = tensor
            else:
                padded[:length, i, ...] = tensor
        return padded, torch.tensor(mask).byte().to(self.device)

    def index(self, item):
        return self._index[item[0], self.inv[item[1]]]

    def invert(self, batch, dim=0):
        return batch.index_select(dim, torch.tensor(self.inv, device=self.device))
