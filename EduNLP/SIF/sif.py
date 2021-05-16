# coding: utf-8
# 2021/5/16 @ tongshiwei

import re
from .meta import SegmentList, TextSegment, QuesMarkSegment, LatexFormulaSegment, FigureSegment


def is_sif(item):
    return True


def to_sif(item):
    return item


def as_sif(item: str, figures: dict = None, safe=True, errors="raise"):
    if safe is True and is_sif(item) is not True:
        item = to_sif(item)

    segments = re.split(r"(\$.+?\$)", item)
    segment_list = SegmentList()
    for segment in segments:
        if not re.match(r"\$.+?\$", segment):
            segment_list.append(TextSegment(segment))
        elif re.match(r"\$FORMULAFIGUREID\{.+?}\$", segment):
            segment_list.append(LatexFormulaSegment(segment))
        elif re.match(r"\$FIGUREID\{.+?}\$", segment):
            segment_list.append(FigureSegment(segment))
        elif re.match(r"\$(SIFBLANK)|(SIFBRAKET)\$", segment):
            segment_list.append(QuesMarkSegment(segment))
        else:
            segment_list.append(LatexFormulaSegment(segment))

    return segment_list
