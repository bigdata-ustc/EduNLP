# coding: utf-8
# 2021/8/3 @ tongshiwei

from copy import deepcopy
import numpy as np


class Masker(object):
    """
    Examples
    -------
    >>> masker = Masker(per=0.5, seed=10)
    >>> items = [[1, 1, 3, 4, 6], [2], [5, 9, 1, 4]]
    >>> masked_seq, mask_label = masker(items)
    >>> masked_seq
    [[1, 1, 0, 0, 6], [2], [0, 9, 0, 4]]
    >>> mask_label
    [[0, 0, 1, 1, 0], [0], [1, 0, 1, 0]]
    >>> items = [[1, 2, 3], [1, 1, 0], [2, 0, 0]]
    >>> masked_seq, mask_label = masker(items, [3, 2, 1])
    >>> masked_seq
    [[1, 0, 3], [0, 1, 0], [2, 0, 0]]
    >>> mask_label
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    >>> masker = Masker(mask="[MASK]", per=0.5, seed=10)
    >>> items = [["a", "b", "c"], ["d", "[PAD]", "[PAD]"], ["hello", "world", "[PAD]"]]
    >>> masked_seq, mask_label = masker(items, length=[3, 1, 2])
    >>> masked_seq
    [['a', '[MASK]', 'c'], ['d', '[PAD]', '[PAD]'], ['hello', '[MASK]', '[PAD]']]
    >>> mask_label
    [[0, 1, 0], [0, 0, 0], [0, 1, 0]]
    """

    def __init__(self, mask: (int, str, ...) = 0, per=0.2, seed=None):
        """

        Parameters
        ----------
        mask: int, str
        per
        seed
        """
        self.seed = np.random.default_rng(seed)
        self.per = per
        self.mask = mask

    def __call__(self, seqs, length=None, *args, **kwargs) -> tuple:
        seqs = deepcopy(seqs)
        masked_list = []
        if length is None:
            length = [len(seq) for seq in seqs]
        for seq, _length in zip(seqs, length):
            masked = self.seed.choice(len(seq) - 1, size=int(_length * self.per), replace=False)
            _masked_list = [0] * len(seq)
            for _masked in masked:
                seq[_masked] = self.mask
                _masked_list[_masked] = 1
            masked_list.append(_masked_list)
        return seqs, masked_list
