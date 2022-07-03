from ..utils import dict2str4sif
from ..SIF import is_sif, to_sif, sif4sci
from ..SIF.segment import seg, SegmentList
from ..Tokenizer import PureTextTokenizer
from ..SIF.tokenization.text import tokenize


class IsSifPipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, inputs):
        print(is_sif(inputs, *self.args, **self.kwargs))
        return inputs


class ToSifPipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, inputs):
        return to_sif(inputs, *self.args, **self.kwargs)


class Dict2Str4SifPipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, inputs):
        return dict2str4sif(inputs, *self.args, **self.kwargs)


class Sif4SciPipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, inputs):
        return sif4sci(inputs, *self.args, **self.kwargs)


class SegPipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, inputs):
        return seg(inputs, *self.args, **self.kwargs)


class SegDescribePipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, inputs: SegmentList):
        print(inputs.describe())
        return inputs


class SegFilterPipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, inputs: SegmentList):
        inputs.filter(*self.args, **self.kwargs)
        return inputs


class TokenizePipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, inputs):
        return tokenize(inputs, *self.args, **self.kwargs)


class PureTextTokenizerPipe:
    def __init__(self, *args, **kwargs):
        self.tokenizer = PureTextTokenizer()
        self.args = args
        self.kwargs = kwargs

    def __call__(self, inputs):
        return [i for i in self.tokenizer(inputs, *self.args, **self.kwargs)]


TOKENIZER_PIPES = {
    'dict2str4sif': Dict2Str4SifPipe,
    'is_sif': IsSifPipe,
    'to_sif': ToSifPipe,
    'sif4sci': Sif4SciPipe,
    'seg': SegPipe,
    'seg_describe': SegDescribePipe,
    'seg_filter': SegFilterPipe,
    'tokenize': TokenizePipe,
    'pure_text_tokenizer': PureTextTokenizerPipe
}
