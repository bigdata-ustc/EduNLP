from ..utils import dict2str4sif
from ..SIF import is_sif, to_sif, sif4sci
from ..SIF.segment import seg, SegmentList
from ..Tokenizer import PureTextTokenizer
from ..SIF.tokenization.text import tokenize


class IsSifPipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input_):
        print(is_sif(input_, *self.args, **self.kwargs))
        return input_


class ToSifPipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input_):
        return to_sif(input_, *self.args, **self.kwargs)


class Dict2Str4SifPipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input_):
        return dict2str4sif(input_, *self.args, **self.kwargs)


class Sif4SciPipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input_):
        return sif4sci(input_, *self.args, **self.kwargs)


class SegPipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input_):
        return seg(input_, *self.args, **self.kwargs)


class SegDescribePipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input_: SegmentList):
        print(input_.describe())
        return input


class SegFilterPipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input_: SegmentList):
        input_.filter(*self.args, **self.kwargs)
        return input_


class TokenizePipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input_):
        return tokenize(input_, *self.args, **self.kwargs)


class PureTextTokenizerPipe:
    def __init__(self, *args, **kwargs):
        self.tokenizer = PureTextTokenizer()
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input_):
        return [i for i in self.tokenizer(input_, *self.args, **self.kwargs)]


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
