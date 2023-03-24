# coding: utf-8
# 2021/7/13 @ tongshiwei
import torch


class Vector(object):
    def infer_vector(self, items, *args, **kwargs) -> ...:
        pass

    def infer_tokens(self, items, *args, **kwargs) -> ...:
        pass

    @property
    def vector_size(self):
        raise NotImplementedError

    @property
    def is_frozen(self):  # pragma: no cover
        return True

    def freeze(self, *args, **kwargs):  # pragma: no cover
        pass

    def cuda_tensor(self, items: dict):
        for k, v in items.items():
            if isinstance(v, torch.Tensor):
                items[k] = v.to(self.device)