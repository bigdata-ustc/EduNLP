# coding: utf-8
# 2021/5/20 @ tongshiwei

import os
from pathlib import PurePath


def abs_current_dir(filepath):
    """
    获取文件所在目录的绝对路径

    Example
    -------
    .. code ::

        abs_current_dir(__file__)

    """
    return os.path.abspath(os.path.dirname(filepath))


def path_append(path, *addition, to_str=False):
    """
    路径合并函数

    Examples
    --------
    .. code-block:: python

    path_append("../", "../data", "../dataset1/", "train", to_str=True)
    '../../data/../dataset1/train'

    Parameters
    ----------
    path: str or PurePath
    addition: list(str or PurePath)
    to_str: bool
        Convert the new path to str
    Returns
    -------

    """
    path = PurePath(path)
    if addition:
        for a in addition:
            path = path / a
    if to_str:
        return str(path)
    return path
