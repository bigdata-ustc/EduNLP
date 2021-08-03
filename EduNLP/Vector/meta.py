# coding: utf-8
# 2021/7/13 @ tongshiwei

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
