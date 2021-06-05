import pytest
from EduNLP.SIF.parser.parser import Parser


def test_parser():
    text = ''
    textparser = Parser(text)
    textparser.description_list()

    text = '随机$text{观测}$生产某种零件的A工厂25名工人的日加工零件数_   _'
    textparser = Parser(text)
    textparser.description_list()

    text = 'X的分布列为(   )'
    textparser = Parser(text)
    textparser.description_list()

    text = '由题意得（ ）'
    textparser = Parser(text)
    textparser.description_list()
    assert textparser.error_flag == 0

    text = '1.命题中真命题的序号是\n ① AB是⊙O的直径，AC是⊙O的切线，BC交⊙O于点E．AC的中点为D'
    textparser = Parser(text)
    textparser.description_list()
    assert textparser.error_flag == 1
