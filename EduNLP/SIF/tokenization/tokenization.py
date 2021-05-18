# coding: utf-8
# 2021/5/18 @ tongshiwei

from ..constants import Symbol, TEXT_SYMBOL, FIGURE_SYMBOL, FORMULA_SYMBOL, QUES_MARK_SYMBOL
from ..segment import (SegmentList, TextSegment, FigureSegment, LatexFormulaSegment, FigureFormulaSegment,
                       QuesMarkSegment)
from . import text, formula


class TokenList(object):
    def __init__(self, segment_list: SegmentList, text_params=None, formula_params=None):
        self._tokens = []
        self._text_tokens = []
        self._formula_tokens = []
        self._figure_tokens = []
        self._ques_mark_tokens = []
        self.text_params = text_params if text_params is not None else {}
        self.formula_params = formula_params if formula_params is not None else {}
        self.extend(segment_list.segments)

    @property
    def tokens(self):
        return self._tokens

    def add(self, *segment):
        for seg in segment:
            self.append(seg)

    def append_text(self, segment, symbol=False):
        if symbol is False:
            tokens = text.tokenize(segment, **self.text_params)
            for token in tokens:
                self._text_tokens.append(len(self._tokens))
                self._tokens.append(token)
        else:
            self._text_tokens.append(len(self._tokens))
            self._tokens.append(segment)

    def append_formula(self, segment, symbol=False):
        if symbol is True or not self.formula_params:
            self._formula_tokens.append(len(self._tokens))
            self._tokens.append(segment)
        else:
            formula_tokens = formula.tokenize(segment, **self.formula_params)
            for token in formula_tokens:
                self._formula_tokens.append(len(self._tokens))
                self._tokens.append(token)

    def append_figure(self, segment, **kwargs):
        self._figure_tokens.append(len(self._tokens))
        self._tokens.append(segment)

    def append_ques_mark(self, segment, **kwargs):
        self._ques_mark_tokens.append(len(self._tokens))
        self._tokens.append(segment)

    def append(self, segment):
        if isinstance(segment, TextSegment):
            self.append_text(segment)
        elif isinstance(segment, (LatexFormulaSegment, FigureFormulaSegment)):
            self.append_formula(segment)
        elif isinstance(segment, FigureSegment):
            self.append_figure(segment)
        elif isinstance(segment, QuesMarkSegment):
            self.append_ques_mark(segment)
        elif isinstance(segment, Symbol):
            if segment == TEXT_SYMBOL:
                self.append_text(segment, symbol=True)
            elif segment == FORMULA_SYMBOL:
                self.append_formula(segment, symbol=True)
            elif segment == FIGURE_SYMBOL:
                self.append_figure(segment, symbol=True)
            elif segment == QUES_MARK_SYMBOL:
                self.append_ques_mark(segment, symbol=True)
            else:
                raise TypeError()
        else:
            raise TypeError()

    def extend(self, segments):
        for segment in segments:
            self.append(segment)


def tokenize(segment_list: SegmentList, text_params=None, formula_params=None):
    return TokenList(segment_list, text_params, formula_params).tokens
