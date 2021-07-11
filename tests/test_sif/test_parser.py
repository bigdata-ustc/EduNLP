import pytest
from EduNLP.SIF.parser.parser import Parser


def test_parser():
    text = ''
    text_parser = Parser(text)
    text_parser.description_list()

    text = '随机$text{观测}$生产某种零件的A工厂25名工人的日加工零件数_   _'
    text_parser = Parser(text)
    text_parser.description_list()

    text = 'X的分布列为(   )'
    text_parser = Parser(text)
    text_parser.description_list()

    text = '由题意得（ ）'
    text_parser = Parser(text)
    text_parser.description_list()
    assert text_parser.error_flag == 0

    text = '1.命题中真命题的序号是\n ① AB是⊙O的直径，AC是⊙O的切线，BC交⊙O于点E．AC的中点为D'
    text_parser = Parser(text)
    text_parser.description_list()
    assert text_parser.error_flag == 1

    text = r"公式两侧的匹配符号需要完整，如不允许$\frac{y}{x}"
    text_parser = Parser(text)
    text_parser.description_list()
    assert text_parser.error_flag == 1

    text = r"支持公式如$\frac{y}{x}$，$\SIFBlank$，$\FigureID{1}$，不支持公式如$\frac{ \dddot y}{x}$"
    text_parser = Parser(text)
    text_parser.description_list()
    assert text_parser.fomula_illegal_flag == 1
