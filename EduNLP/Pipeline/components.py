from ..utils import dict2str4sif
from ..SIF import is_sif, to_sif, sif4sci
from ..SIF.segment import seg, SegmentList
from ..Tokenizer import PureTextTokenizer
from ..SIF.tokenization.text import tokenize


class BasePipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input_):
        raise NotImplementedError


class IsSifPipe(BasePipe):
    def __init__(self, *args, **kwargs):
        super(IsSifPipe, self).__init__(*args, **kwargs)

    def __call__(self, input_):
        print(is_sif(input_, *self.args, **self.kwargs))
        return input_


class ToSifPipe(BasePipe):
    def __init__(self, *args, **kwargs):
        super(ToSifPipe, self).__init__(*args, **kwargs)

    def __call__(self, input_):
        return to_sif(input_, *self.args, **self.kwargs)


class Dict2Str4SifPipe(BasePipe):
    def __init__(self, *args, **kwargs):
        super(Dict2Str4SifPipe, self).__init__(*args, **kwargs)

    def __call__(self, input_):
        return dict2str4sif(input_, *self.args, **self.kwargs)


class Sif4SciPipe(BasePipe):
    def __init__(self, *args, **kwargs):
        super(Sif4SciPipe, self).__init__(*args, **kwargs)

    def __call__(self, input_):
        return sif4sci(input_, *self.args, **self.kwargs)


class SegPipe(BasePipe):
    def __init__(self, *args, **kwargs):
        super(SegPipe, self).__init__(*args, **kwargs)

    def __call__(self, input_):
        return seg(input_, *self.args, **self.kwargs)


class SegDescribePipe(BasePipe):
    def __init__(self, *args, **kwargs):
        super(SegDescribePipe, self).__init__(*args, **kwargs)

    def __call__(self, input_: SegmentList):
        print(input_.describe())
        return input


class SegFilterPipe(BasePipe):
    def __init__(self, *args, **kwargs):
        super(SegFilterPipe, self).__init__(*args, **kwargs)

    def __call__(self, input_: SegmentList):
        input_.filter(*self.args, **self.kwargs)
        return input_


class TokenizePipe(BasePipe):
    def __init__(self, *args, **kwargs):
        super(TokenizePipe, self).__init__(*args, **kwargs)

    def __call__(self, input_):
        return tokenize(input_, *self.args, **self.kwargs)


class PureTextTokenizerPipe(BasePipe):
    def __init__(self, *args, **kwargs):
        super(PureTextTokenizerPipe, self).__init__(*args, **kwargs)
        self.tokenizer = PureTextTokenizer()

    def __call__(self, input_):
        return [i for i in self.tokenizer(input_, *self.args, **self.kwargs)]


PREPROCESSING_PIPES = {
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
