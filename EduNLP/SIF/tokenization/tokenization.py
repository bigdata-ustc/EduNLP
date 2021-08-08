# coding: utf-8
# 2021/5/18 @ tongshiwei

from contextlib import contextmanager
from EduNLP.Formula import link_formulas as _link_formulas, Formula
from ..constants import (
    Symbol, TEXT_SYMBOL, FIGURE_SYMBOL, FORMULA_SYMBOL, QUES_MARK_SYMBOL, TAG_SYMBOL, SEP_SYMBOL,
    TEXT_BEGIN, TEXT_END, FORMULA_BEGIN, FORMULA_END
)
from ..segment import (SegmentList, TextSegment, FigureSegment, LatexFormulaSegment, FigureFormulaSegment,
                       QuesMarkSegment, Figure, TagSegment, SepSegment)
from . import text, formula

__all__ = ["TokenList", "tokenize", "link_formulas"]


class TokenList(object):
    """
    Attributes
    -------------

    """
    def __init__(self, segment_list: SegmentList, text_params=None, formula_params=None, figure_params=None):
        self._tokens = []
        self._text_tokens = []
        self._formula_tokens = []
        self._figure_tokens = []
        self._ques_mark_tokens = []
        self._tag_tokens = []
        self._sep_tokens = []
        self._segments = []
        self._seg_types = {
            "t": [],
            "f": [],
            "g": [],
            "m": [],
            "a": [],
            "s": []
        }
        self.text_params = text_params if text_params is not None else {}
        if formula_params is not None and "symbolize_figure_formula" in formula_params:
            self.symbolize_figure_formula = formula_params.pop("symbolize_figure_formula")
        else:
            self.symbolize_figure_formula = False
        self.formula_params = formula_params if formula_params is not None else {"method": "linear"}
        self.formula_tokenize_method = self.formula_params.get("method")
        self.figure_params = figure_params if figure_params is not None else {}
        self.extend(segment_list.segments)
        self._token_idx = None

    def _variable_standardization(self):
        if self.formula_tokenize_method == "ast":
            ast_formulas = [self._tokens[i] for i in self._formula_tokens if isinstance(self._tokens[i], Formula)]
            if ast_formulas:
                _link_formulas(*ast_formulas, link_vars=self.formula_params.get("var_numbering", False))

    @contextmanager
    def add_seg_type(self, seg_type, tar: list, add_seg_type=True, mode="delimiter"):
        if add_seg_type is True and mode in {"delimiter", "head"}:
            if seg_type == "t":
                tar.append(TEXT_BEGIN)
            elif seg_type == "f" and (
                    self.formula_params.get("method") == "ast" and self.formula_params.get("return_type", "list")
            ):
                tar.append(FORMULA_BEGIN)
        yield
        if add_seg_type is True and mode in {"delimiter", "tail"}:
            if seg_type == "t":
                tar.append(TEXT_END)
            elif seg_type == "f" and (
                    self.formula_params.get("method") == "ast" and self.formula_params.get("return_type", "list")
            ):
                tar.append(FORMULA_END)

    def get_segments(self, add_seg_type=True, add_seg_mode="delimiter", keep="*", drop="",
                     depth=None):  # pragma: no cover
        r"""

        Parameters
        ----------
        add_seg_type
        add_seg_mode:
            delimiter: both in the head and at the tail
            head: only in the head
            tail: only at the tail
        keep
        drop
        depth: int or None
            0: only separate at \SIFSep
            1: only separate at \SIFTag
            2: separate at \SIFTag and \SIFSep
            otherwise, separate all segments

        Returns
        -------

        """
        keep = set("tfgmas" if keep == "*" else keep) - set(drop)
        _segments = []
        _segment = []
        close_tag = False
        for start, end, seg_type in self._segments:
            if depth == 0:
                if seg_type == "s":
                    close_tag = True
            elif depth == 1:
                if seg_type == "a":
                    close_tag = True
            elif depth == 2:
                if seg_type in {"s", "a"}:
                    close_tag = True
            else:
                close_tag = True
            if seg_type in keep:
                with self.add_seg_type(seg_type, _segment, add_seg_type, add_seg_mode):
                    for token in self._tokens[start: end]:
                        self.__add_token(token, _segment)
            if close_tag is True and _segment:
                _segments.append(_segment)
                _segment = []
        return _segments

    def __get_segments(self, seg_type):
        _segments = []
        for i in self._seg_types[seg_type]:
            _segment = []
            start, end, _ = self._segments[i]
            for token in self._tokens[start: end]:
                self.__add_token(token, _segment)
            if _segment:
                _segments.append(_segment)
        return _segments

    @property
    def text_segments(self):
        return self.__get_segments("t")

    @property
    def formula_segments(self):
        return self.__get_segments("f")

    @property
    def figure_segments(self):
        return self.__get_segments("g")

    @property
    def ques_mark_segments(self):
        return self.__get_segments("m")

    @property
    def tokens(self):
        tokens = []
        if self._token_idx is not None:
            for i, token in enumerate(self._tokens):
                if i in self._token_idx:
                    self.__add_token(token, tokens)
        else:
            for token in self._tokens:
                self.__add_token(token, tokens)
        return tokens

    def append_text(self, segment, symbol=False):
        with self._append("t"):
            if symbol is False:
                tokens = text.tokenize(segment, **self.text_params)
                for token in tokens:
                    self._text_tokens.append(len(self._tokens))
                    self._tokens.append(token)
            else:
                self._text_tokens.append(len(self._tokens))
                self._tokens.append(segment)

    def append_formula(self, segment, symbol=False, init=True):
        with self._append("f"):
            if symbol is True:
                self._formula_tokens.append(len(self._tokens))
                self._tokens.append(segment)
            elif self.symbolize_figure_formula and isinstance(segment, FigureFormulaSegment):
                self._formula_tokens.append(len(self._tokens))
                self._tokens.append(Symbol(FORMULA_SYMBOL))
            elif isinstance(segment, FigureFormulaSegment):
                self._formula_tokens.append(len(self._tokens))
                self._tokens.append(segment)
            elif self.formula_params.get("method") == "ast":
                self._formula_tokens.append(len(self._tokens))
                self._tokens.append(Formula(segment, init=init))
            else:
                tokens = formula.tokenize(segment, **self.formula_params)
                for token in tokens:
                    self._formula_tokens.append(len(self._tokens))
                    self._tokens.append(token)

    def append_figure(self, segment, **kwargs):
        with self._append("g"):
            self._figure_tokens.append(len(self._tokens))
            self._tokens.append(segment)

    def append_ques_mark(self, segment, **kwargs):
        with self._append("m"):
            self._ques_mark_tokens.append(len(self._tokens))
            self._tokens.append(segment)

    def append_tag(self, segment, **kwargs):
        with self._append("a"):
            self._tag_tokens.append(len(self._tokens))
            self._tokens.append(segment)

    def append_sep(self, segment, **kwargs):
        with self._append("s"):
            self._sep_tokens.append(len(self._tokens))
            self._tokens.append(segment)

    @contextmanager
    def _append(self, seg_type):
        start = len(self._tokens)
        yield
        end = len(self._tokens)
        self._seg_types[seg_type].append(len(self._segments))
        self._segments.append((start, end, seg_type))

    def append(self, segment, lazy=False):
        if isinstance(segment, TextSegment):
            self.append_text(segment)
        elif isinstance(segment, (LatexFormulaSegment, FigureFormulaSegment)):
            self.append_formula(segment, init=not lazy)
            if lazy is False:
                self._variable_standardization()
        elif isinstance(segment, FigureSegment):
            self.append_figure(segment)
        elif isinstance(segment, QuesMarkSegment):
            self.append_ques_mark(segment)
        elif isinstance(segment, TagSegment):
            self.append_tag(segment)
        elif isinstance(segment, SepSegment):
            self.append_sep(segment)
        elif isinstance(segment, Symbol):
            if segment == TEXT_SYMBOL:
                self.append_text(segment, symbol=True)
            elif segment == FORMULA_SYMBOL:
                self.append_formula(segment, symbol=True)
            elif segment == FIGURE_SYMBOL:
                self.append_figure(segment, symbol=True)
            elif segment == QUES_MARK_SYMBOL:
                self.append_ques_mark(segment, symbol=True)
            elif segment == TAG_SYMBOL:
                self.append_tag(segment, symbol=True)
            elif segment == SEP_SYMBOL:
                self.append_sep(segment, symbol=True)
            else:
                raise TypeError("Unknown symbol type: %s" % segment)
        else:
            raise TypeError("Unknown segment type: %s" % type(segment))

    def extend(self, segments):
        for segment in segments:
            self.append(segment, True)
        self._variable_standardization()

    @property
    def text_tokens(self):
        return [self._tokens[i] for i in self._text_tokens]

    def __add_token(self, token, tokens):
        if isinstance(token, Formula):
            if self.formula_params.get("return_type") == "list":
                tokens.extend(formula.traversal_formula(token.ast_graph, **self.formula_params))
            elif self.formula_params.get("return_type") == "ast":
                tokens.append(token.ast_graph)
            else:
                tokens.append(token)
        elif isinstance(token, Figure):
            if self.figure_params.get("figure_instance") is True:
                tokens.append(token.figure)
            else:
                tokens.append(token)
        else:
            tokens.append(token)

    @property
    def formula_tokens(self):
        tokens = []
        for i in self._formula_tokens:
            self.__add_token(self._tokens[i], tokens)
        return tokens

    @property
    def figure_tokens(self):
        tokens = []
        for i in self._figure_tokens:
            self.__add_token(self._tokens[i], tokens)
        return tokens

    @property
    def ques_mark_tokens(self):
        return [self._tokens[i] for i in self._ques_mark_tokens]

    def __repr__(self):
        return str(self.tokens)

    @property
    def inner_formula_tokens(self):
        return [self._tokens[i] for i in self._formula_tokens]

    @contextmanager
    def filter(self, drop: (set, str) = "", keep: (set, str) = "*"):
        _drop = {c for c in drop} if isinstance(drop, str) else drop
        if keep == "*":
            _keep = {c for c in "tfgmas" if c not in _drop}
        else:
            _keep = {c for c in keep if c not in _drop} if isinstance(keep, str) else keep
        self._token_idx = set()
        if "t" in _keep:
            self._token_idx |= set(self._text_tokens)
        if "f" in _keep:
            self._token_idx |= set(self._formula_tokens)
        if "g" in _keep:
            self._token_idx |= set(self._figure_tokens)
        if "m" in _keep:
            self._token_idx |= set(self._ques_mark_tokens)
        if "a" in _keep:
            self._token_idx |= set(self._tag_tokens)
        if "s" in _keep:
            self._token_idx |= set(self._sep_tokens)
        yield
        self._token_idx = None

    def describe(self):
        return {
            "t": len(self._text_tokens),
            "f": len(self._formula_tokens),
            "g": len(self._figure_tokens),
            "m": len(self._ques_mark_tokens),
        }


def tokenize(segment_list: SegmentList, text_params=None, formula_params=None, figure_params=None):
    return TokenList(segment_list, text_params, formula_params, figure_params)


def link_formulas(*token_list: TokenList, link_vars=True):
    ast_formulas = []
    for tl in token_list:
        if tl.formula_tokenize_method == "ast":
            ast_formulas.extend([token for token in tl.inner_formula_tokens if isinstance(token, Formula)])
    _link_formulas(*ast_formulas, link_vars=link_vars)
