# coding: utf-8
# 2021/5/29 @ tongshiwei
import logging


def get_logger():
    _logger = logging.getLogger("EduNLP")
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(name)s, %(levelname)s %(message)s'))
    ch.setLevel(logging.INFO)
    _logger.addHandler(ch)
    return _logger


logger = get_logger()
