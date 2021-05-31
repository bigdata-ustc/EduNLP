# coding: utf-8
# 2021/5/18 @ tongshiwei
import base64
import numpy as np
import re
from contextlib import contextmanager
from ..constants import Symbol, TEXT_SYMBOL, FORMULA_SYMBOL, FIGURE_SYMBOL, QUES_MARK_SYMBOL, TAG_SYMBOL, SEP_SYMBOL


class TextSegment(str):
    pass


class LatexFormulaSegment(str):
    pass


class Figure(object):
    def __init__(self, is_base64=False):
        self.base64 = is_base64
        self.figure = None

    @classmethod
    def base64_to_numpy(cls, figure: str):
        return np.frombuffer(base64.b64decode(figure), dtype=np.uint8)


class FigureFormulaSegment(Figure):
    def __init__(self, src, is_base64=False, figure_instance: (dict, bool) = None):
        super(FigureFormulaSegment, self).__init__(is_base64)
        self.src = src
        if self.base64 is True:
            self.figure = self.src[len(r"\FormFigureBase64") + 1: -1]
            if figure_instance is True or (isinstance(figure_instance, dict) and figure_instance.get("base64") is True):
                self.figure = self.base64_to_numpy(self.figure)
        else:
            self.figure = self.src[len(r"\FormFigureID") + 1: -1]
            if isinstance(figure_instance, dict):
                self.figure = figure_instance[self.figure]

    def __repr__(self):
        if self.base64 is True:
            return FORMULA_SYMBOL
        return str(self.src)


class FigureSegment(Figure):
    def __init__(self, src, is_base64=False, figure_instance: (dict, bool) = None):
        super(FigureSegment, self).__init__(is_base64)
        self.src = src
        if self.base64 is True:
            self.figure = self.src[len(r"\FigureBase64") + 1: -1]
            if figure_instance is True or (isinstance(figure_instance, dict) and figure_instance.get("base64") is True):
                self.figure = self.base64_to_numpy(self.figure)
        else:
            self.figure = self.src[len(r"\FigureID") + 1: -1]
            if isinstance(figure_instance, dict):
                self.figure = figure_instance[self.figure]

    def __repr__(self):
        if self.base64 is True:
            return FIGURE_SYMBOL
        return str(self.src)


class QuesMarkSegment(str):
    pass


class TagSegment(str):
    pass


class SepSegment(str):
    pass


class SegmentList(object):
    def __init__(self, item, figures: dict = None):
        self._segments = []
        self._text_segments = []
        self._formula_segments = []
        self._figure_segments = []
        self._ques_mark_segments = []
        self._tag_segments = []
        self._sep_segments = []
        segments = re.split(r"(\$.+?\$)", item)
        for segment in segments:
            if not segment:
                continue
            if not re.match(r"\$.+?\$", segment):
                self.append(TextSegment(segment))
            elif re.match(r"\$\\FormFigureID\{.+?}\$", segment):
                self.append(FigureFormulaSegment(segment[1:-1], is_base64=False, figure_instance=figures))
            elif re.match(r"\$\\FormFigureBase64\{.+?}\$", segment):
                self.append(FigureFormulaSegment(segment[1:-1], is_base64=True, figure_instance=figures))
            elif re.match(r"\$\\FigureID\{.+?}\$", segment):
                self.append(FigureSegment(segment[1:-1], is_base64=False, figure_instance=figures))
            elif re.match(r"\$\\FigureBase64\{.+?}\$", segment):
                self.append(FigureSegment(segment[1:-1], is_base64=True, figure_instance=figures))
            elif re.match(r"\$\\(SIFBlank|SIFChoice)\$", segment):
                self.append(QuesMarkSegment(segment[1:-1]))
            elif re.match(r"\$\\SIFTag\{.+?}\$", segment):
                self.append(TagSegment(segment[1:-1]))
            elif re.match(r"\$\\SIFSep\$", segment):
                self.append(SepSegment(segment[1:-1]))
            else:
                self.append(LatexFormulaSegment(segment[1:-1]))
        self._seg_idx = None

    def __repr__(self):
        return str(self.segments)

    def __len__(self):
        return len(self._segments)

    def append(self, segment) -> None:
        if isinstance(segment, TextSegment):
            self._text_segments.append(len(self))
        elif isinstance(segment, (LatexFormulaSegment, FigureFormulaSegment)):
            self._formula_segments.append(len(self))
        elif isinstance(segment, FigureSegment):
            self._figure_segments.append(len(self))
        elif isinstance(segment, QuesMarkSegment):
            self._ques_mark_segments.append(len(self))
        elif isinstance(segment, TagSegment):
            self._tag_segments.append(len(self))
        elif isinstance(segment, SepSegment):
            self._sep_segments.append(len(self))
        else:
            raise TypeError("Unknown Segment Type: %s" % type(segment))
        self._segments.append(segment)

    @property
    def segments(self):
        if self._seg_idx is None:
            return self._segments
        else:
            return [s for i, s in enumerate(self._segments) if i in self._seg_idx]

    @property
    def text_segments(self):
        return [self._segments[i] for i in self._text_segments]

    @property
    def formula_segments(self):
        return [self._segments[i] for i in self._formula_segments]

    @property
    def figure_segments(self):
        return [self._segments[i] for i in self._figure_segments]

    @property
    def ques_mark_segments(self):
        return [self._segments[i] for i in self._ques_mark_segments]

    @property
    def tag_segments(self):
        return [self._segments[i] for i in self._tag_segments]

    def to_symbol(self, idx, symbol):
        self._segments[idx] = symbol

    def symbolize(self, to_symbolize="fgm"):
        """

        Parameters
        ----------
        to_symbolize:
            "t": text
            "f": formula
            "g": figure
            "m": question mark

        Returns
        -------

        """
        if "t" in to_symbolize:
            for idx in self._text_segments:
                self.to_symbol(idx, Symbol(TEXT_SYMBOL))
        if "f" in to_symbolize:
            for idx in self._formula_segments:
                self.to_symbol(idx, Symbol(FORMULA_SYMBOL))
        if "g" in to_symbolize:
            for idx in self._figure_segments:
                self.to_symbol(idx, Symbol(FIGURE_SYMBOL))
        if "m" in to_symbolize:
            for idx in self._ques_mark_segments:
                self.to_symbol(idx, Symbol(QUES_MARK_SYMBOL))
        if "a" in to_symbolize:
            for idx in self._tag_segments:
                self.to_symbol(idx, Symbol(TAG_SYMBOL))
        if "s" in to_symbolize:
            for idx in self._sep_segments:
                self.to_symbol(idx, Symbol(SEP_SYMBOL))

    @contextmanager
    def filter(self, drop: (set, str) = "", keep: (set, str) = "*"):
        _drop = {c for c in drop} if isinstance(drop, str) else drop
        if keep == "*":
            _keep = {c for c in "tfgmas" if c not in _drop}
        else:
            _keep = {c for c in keep if c not in _drop} if isinstance(keep, str) else keep
        self._seg_idx = set()
        if "t" in _keep:
            self._seg_idx |= set(self._text_segments)
        if "f" in _keep:
            self._seg_idx |= set(self._formula_segments)
        if "g" in _keep:
            self._seg_idx |= set(self._figure_segments)
        if "m" in _keep:
            self._seg_idx |= set(self._ques_mark_segments)
        if "a" in _keep:
            self._seg_idx |= set(self._tag_segments)
        if "s" in _keep:
            self._seg_idx |= set(self._sep_segments)
        yield
        self._seg_idx = None

    def describe(self):
        return {
            "t": len(self._text_segments),
            "f": len(self._formula_segments),
            "g": len(self._figure_segments),
            "m": len(self._ques_mark_segments),
        }


def seg(item, figures=None, symbol=None):
    r"""

    Parameters
    ----------
    item
    figures
    symbol

    Returns
    -------

    Examples
    --------
    >>> test_item = r"如图所示，则$\bigtriangleup ABC$的面积是$\SIFBlank$。$\FigureID{1}$"
    >>> s = seg(test_item)
    >>> s
    ['如图所示，则', '\\bigtriangleup ABC', '的面积是', '\\SIFBlank', '。', \FigureID{1}]
    >>> s.describe()
    {'t': 3, 'f': 1, 'g': 1, 'm': 1}
    >>> with s.filter("f"):
    ...     s
    ['如图所示，则', '的面积是', '\\SIFBlank', '。', \FigureID{1}]
    >>> with s.filter(keep="t"):
    ...     s
    ['如图所示，则', '的面积是', '。']
    >>> with s.filter():
    ...     s
    ['如图所示，则', '\\bigtriangleup ABC', '的面积是', '\\SIFBlank', '。', \FigureID{1}]
    >>> seg(test_item, symbol="fgm")
    ['如图所示，则', '[FORMULA]', '的面积是', '[MARK]', '。', '[FIGURE]']
    >>> seg(test_item, symbol="tfgm")
    ['[TEXT]', '[FORMULA]', '[TEXT]', '[MARK]', '[TEXT]', '[FIGURE]']
    >>> seg(r"如图所示，则$\FormFigureID{0}$的面积是$\SIFBlank$。$\FigureID{1}$")
    ['如图所示，则', \FormFigureID{0}, '的面积是', '\\SIFBlank', '。', \FigureID{1}]
    >>> seg(r"如图所示，则$\FormFigureID{0}$的面积是$\SIFBlank$。$\FigureID{1}$", symbol="fgm")
    ['如图所示，则', '[FORMULA]', '的面积是', '[MARK]', '。', '[FIGURE]']
    >>> s.text_segments
    ['如图所示，则', '的面积是', '。']
    >>> s.formula_segments
    ['\\bigtriangleup ABC']
    >>> s.figure_segments
    [\FigureID{1}]
    >>> s.ques_mark_segments
    ['\\SIFBlank']
    >>> test_item_1 = {
    ...     "stem": r"若复数$z=1+2 i+i^{3}$，则$|z|=$",
    ...     "options": ['0', '1', r'$\sqrt{2}$', '2']
    ... }
    >>> from EduNLP.utils import dict2str4sif
    >>> test_item_1_str = dict2str4sif(test_item_1)
    >>> test_item_1_str  # doctest: +ELLIPSIS
    '$\\SIFTag{stem_begin}$...$\\SIFTag{stem_end}$$\\SIFTag{options_begin}$$\\SIFTag{list_0}$0...$\\SIFTag{options_end}$'
    >>> s1 = seg(test_item_1_str, symbol="tfgm")
    >>> s1  # doctest: +ELLIPSIS
    ['\\SIFTag{stem_begin}'...'\\SIFTag{stem_end}', '\\SIFTag{options_begin}', '\\SIFTag{list_0}', ...]
    >>> with s1.filter(keep="a"):
    ...     s1  # doctest: +ELLIPSIS
    [...'\\SIFTag{list_0}', '\\SIFTag{list_1}', '\\SIFTag{list_2}', '\\SIFTag{list_3}', '\\SIFTag{options_end}']
    >>> s1.tag_segments  # doctest: +ELLIPSIS
    ['\\SIFTag{stem_begin}', '\\SIFTag{stem_end}', '\\SIFTag{options_begin}', ... '\\SIFTag{options_end}']
    >>> test_item_1_str_2 = dict2str4sif(test_item_1, tag_mode="head", add_list_no_tag=False)
    >>> seg(test_item_1_str_2, symbol="tfgmas")  # doctest: +ELLIPSIS
    ['[TAG]', ... '[TAG]', '[TEXT]', '[SEP]', '[TEXT]', '[SEP]', '[FORMULA]', '[SEP]', '[TEXT]']
    >>> s2 = seg(test_item_1_str_2, symbol="fgm")
    >>> s2.tag_segments
    ['\\SIFTag{stem}', '\\SIFTag{options}']
    """
    segments = SegmentList(item, figures)
    if symbol is not None:
        segments.symbolize(symbol)
    return segments
