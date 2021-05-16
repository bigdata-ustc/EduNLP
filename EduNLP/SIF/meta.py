# coding: utf-8
# 2021/5/16 @ tongshiwei


class SegmentList(list):
    def __init__(self, *args, **kwargs):
        super(SegmentList, self).__init__(*args, **kwargs)

        self._text_segments = []
        self._formula_segments = []
        self._figure_segments = []
        self._ques_mark_segments = []

    def append(self, seg) -> None:
        self.append(seg)
        if isinstance(seg, TextSegment):
            self._text_segments.append(len(self))
        elif isinstance(seg, (LatexFormulaSegment, FigureFormulaSegment)):
            self._formula_segments.append(len(self))
        elif isinstance(seg, FigureSegment):
            self._figure_segments.append(len(self))
        elif isinstance(seg, QuesMarkSegment):
            self._ques_mark_segments.append(len(self))
        else:
            raise TypeError("Unknown Segment Type: %s" % type(seg))


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
