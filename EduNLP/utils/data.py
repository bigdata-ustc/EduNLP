# coding: utf-8
# 2021/5/30 @ tongshiwei

from contextlib import contextmanager

ann_format = r"$\SIFTag{{{}}}$"
ann_begin_format = r"$\SIFTag{{{}_begin}}$"
ann_end_format = r"$\SIFTag{{{}_end}}$"
ann_list_no_format = r"$\SIFTag{{list_{}}}$"


@contextmanager
def add_annotation(key, tag_mode, tar: list, key_as_tag=True):
    if key_as_tag is True:
        if tag_mode == "delimiter":
            tar.append(ann_begin_format.format(key))
        elif tag_mode == "head":
            tar.append(ann_format.format(key))
    yield
    if key_as_tag is True:
        if tag_mode == "delimiter":
            tar.append(ann_end_format.format(key))
        elif tag_mode == "tail":
            tar.append(ann_format.format(key))


def dict2str4sif(obj: dict, key_as_tag=True, tag_mode="delimiter", add_list_no_tag=True, keys=None) -> str:
    r"""

    Parameters
    ----------
    obj
    key_as_tag
    tag_mode
        delimiter: add $\SIFTag{key_begin}$ in the head and add $\SIFTag{key_end}$ at the end
        head: add $\SIFTag{key}$ in the head
        tail: add $\SIFTag{key}$ at the end
    add_list_no_tag
    keys

    Returns
    -------
    >>> item = {
    ...     "stem": r"若复数$z=1+2 i+i^{3}$，则$|z|=$",
    ...     "options": ['0', '1', r'$\sqrt{2}$', '2'],
    ... }
    >>> item
    {'stem': '若复数$z=1+2 i+i^{3}$，则$|z|=$', 'options': ['0', '1', '$\\sqrt{2}$', '2']}
    >>> dict2str4sif(item) # doctest: +ELLIPSIS
    '$\\SIFTag{stem_begin}$...$\\SIFTag{stem_end}$$\\SIFTag{options_begin}$...$\\SIFTag{options_end}$'
    >>> dict2str4sif(item, add_list_no_tag=True) # doctest: +ELLIPSIS
    '...$\\SIFTag{options_begin}$$\\SIFTag{list_0}$0$\\SIFTag{list_1}$1...$\\SIFTag{options_end}$'
    >>> dict2str4sif(item, tag_mode="head") # doctest: +ELLIPSIS
    '$\\SIFTag{stem}$...$\\SIFTag{options}$...'
    >>> dict2str4sif(item, tag_mode="tail") # doctest: +ELLIPSIS
    '若复数$z=1+2 i+i^{3}$，则$|z|=$$\\SIFTag{stem}$...2$\\SIFTag{options}$'
    >>> dict2str4sif(item, add_list_no_tag=False) # doctest: +ELLIPSIS
    '...$\\SIFTag{options_begin}$0$\\SIFSep$1$\\SIFSep$...$\\SIFTag{options_end}$'
    >>> dict2str4sif(item, key_as_tag=False)
    '若复数$z=1+2 i+i^{3}$，则$|z|=$0$\\SIFSep$1$\\SIFSep$$\\sqrt{2}$$\\SIFSep$2'
    """
    ret = []
    keys = obj.keys() if keys is None else keys
    for key in keys:
        _obj = []
        value = obj[key]
        with add_annotation(key, tag_mode, _obj, key_as_tag):
            if isinstance(value, str):
                _obj.append(value)
            elif isinstance(value, (list, dict)):
                value = value.values() if isinstance(value, dict) else value
                for i, v in enumerate(value):
                    v = str(v)
                    if key_as_tag is True and add_list_no_tag is True:
                        _obj.append(ann_list_no_format.format(i))
                    else:
                        if i > 0:
                            _obj.append(r"$\SIFSep$")
                    _obj.append(v)
            else:  # pragma: no cover
                raise TypeError("Cannot handle %s" % type(value))
        ret.append("".join(_obj))
    return str("".join(ret))
