# coding: utf-8
# 2021/5/30 @ tongshiwei

from contextlib import contextmanager

ann_format = r"$\SIFTag{{{}}}$"
ann_begin_format = r"$\SIFTag{{{}_begin}}$"
ann_end_format = r"$\SIFTag{{{}_end}}$"
ann_list_no_format = r"$\SIFTag{{list_{}}}$"


@contextmanager
def add_annotation(key, annotation_mode, tar: list, key_as_annotation=True):
    if key_as_annotation is True:
        if annotation_mode == "delimiter":
            tar.append(ann_begin_format.format(key))
        elif annotation_mode == "head":
            tar.append(ann_format.format(key))
    yield
    if key_as_annotation is True:
        if annotation_mode == "delimiter":
            tar.append(ann_end_format.format(key))
        elif annotation_mode == "tail":
            tar.append(ann_format.format(key))


def dict2str4sif(obj: dict, key_as_annotation=True, annotation_mode="delimiter", add_list_no_annotation=True) -> str:
    r"""

    Parameters
    ----------
    obj
    key_as_annotation
    annotation_mode
        delimiter: add $\SIFAnn{key_begin}$ in the head and add $\SIFAnn{key_end}$ at the end
        head: add $\SIFAnn{key}$ in the head
        tail: add $\SIFAnn{key}$ at the end
    add_list_no_annotation

    Returns
    -------
    >>> item = {
    ...     "stem": r"若复数$z=1+2 i+i^{3}$，则$|z|=$",
    ...     "options": ['0', '1', '$\sqrt{2}$', '2'],
    ... }
    >>> item
    {'stem': '若复数$z=1+2 i+i^{3}$，则$|z|=$', 'options': ['0', '1', '$\\sqrt{2}$', '2']}
    >>> dict2str4sif(item) # doctest: +ELLIPSIS
    '$\\SIFAnn{stem_begin}$...$\\SIFAnn{stem_end}$$\\SIFAnn{options_begin}$...$\\SIFAnn{options_end}$'
    >>> dict2str4sif(item, add_list_no_annotation=True) # doctest: +ELLIPSIS
    '...$\\SIFAnn{options_begin}$$\\SIFAnn{list_0}$0$\\SIFAnn{list_1}$1...$\\SIFAnn{options_end}$'
    >>> dict2str4sif(item, annotation_mode="head") # doctest: +ELLIPSIS
    '$\\SIFAnn{stem}$...$\\SIFAnn{options}$...'
    >>> dict2str4sif(item, annotation_mode="tail") # doctest: +ELLIPSIS
    '若复数$z=1+2 i+i^{3}$，则$|z|=$$\\SIFAnn{stem}$...2$\\SIFAnn{options}$'
    >>> dict2str4sif(item, add_list_no_annotation=False) # doctest: +ELLIPSIS
    '$\\SIFAnn{stem_begin}$...，则$|z|=$$\\SIFAnn{stem_end}$$\\SIFAnn{options_begin}$0,1,...$\\SIFAnn{options_end}$'
    >>> dict2str4sif(item, key_as_annotation=False)
    '若复数$z=1+2 i+i^{3}$，则$|z|=$0,1,$\\sqrt{2}$,2'
    """
    ret = []

    for key, value in obj.items():
        _obj = []
        with add_annotation(key, annotation_mode, _obj, key_as_annotation):
            if isinstance(value, str):
                _obj.append(value)
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    v = str(v)
                    if key_as_annotation is True and add_list_no_annotation is True:
                        _obj.append(ann_list_no_format.format(i))
                    else:
                        if i > 0:
                            _obj.append(",")
                    _obj.append(v)
            else:
                raise TypeError("Cannot handle %s" % type(value))
        ret.append("".join(_obj))
    return str("".join(ret))
