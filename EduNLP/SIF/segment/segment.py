# coding: utf-8
# 2021/5/18 @ tongshiwei
import re
from ..constants import Symbol, TEXT_SYMBOL, FORMULA_SYMBOL, FIGURE_SYMBOL, QUES_MARK_SYMBOL


class TextSegment(str):
    pass


class LatexFormulaSegment(str):
    pass


class FigureFormulaSegment(str):
    pass


class FigureSegment(str):
    pass


class QuesMarkSegment(str):
    pass


class SegmentList(object):
    def __init__(self, item):
        self._segments = []
        self._text_segments = []
        self._formula_segments = []
        self._figure_segments = []
        self._ques_mark_segments = []
        segments = re.split(r"(\$.+?\$)", item)
        for segment in segments:
            if not segment:
                continue
            if not re.match(r"\$.+?\$", segment):
                self.append(TextSegment(segment))
            elif re.match(r"\$FormFigureID\{.+?}\$", segment):
                self.append(LatexFormulaSegment(segment[1:-1]))
            elif re.match(r"\$FigureID\{.+?}\$", segment):
                self.append(FigureSegment(segment[1:-1]))
            elif re.match(r"\$\\(SIFBlank)|(SIFChoice)\$", segment):
                self.append(QuesMarkSegment(segment[1:-1]))
            else:
                self.append(LatexFormulaSegment(segment[1:-1]))

    def __repr__(self):
        return str(self._segments)

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
        else:
            raise TypeError("Unknown Segment Type: %s" % type(segment))
        self._segments.append(segment)

    @property
    def segments(self):
        return self._segments

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


def seg(item, symbol=None):
    segments = SegmentList(item)
    if symbol is not None:
        segments.symbolize(symbol)
    return segments
