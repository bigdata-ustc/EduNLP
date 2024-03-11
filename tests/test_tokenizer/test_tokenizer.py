# coding: utf-8
# 2021/8/1 @ tongshiwei

import pytest
from EduNLP.Tokenizer import get_tokenizer
from EduNLP.Pretrain import DisenQTokenizer
from EduNLP.utils import abs_current_dir, path_append


def test_tokenizer():
    with pytest.raises(KeyError):
        get_tokenizer("error")


def test_disenQTokenizer():
    tokenizer = DisenQTokenizer(max_length=10, tokenize_method="space")
    # with pytest.raises(RuntimeError):
    #     tokenizer("10 米 的 (2/5) = () 米 的 (1/2) .")

    test_items = [
        "10 米 的 (2/5) = () 米 的 (1/2) . 多 余 的 字",
        "-1 - 1",
        "5 % 2 + 3.14",
        "3.x",
        ".",
        "",
        "-1/2",
        "/",
        "1.2%",
    ]
    tokenizer.set_vocab(test_items)
    print(tokenizer.vocab_size)
    for item in test_items:
        token_item = tokenizer(item)
        print(token_item)

    test_item = tokenizer(test_items[0], padding=True)
    assert test_item["seq_idx"].shape[-1] == 10


def test_CharTokenizer():
    items = [{
        "stem": "文具店有 $600$ 本练习本，卖出一些后，还剩 $4$ 包，每包 $25$ 本，卖出多少本？",
        "options": ["1", "2"]
    }]
    tokenizer = get_tokenizer("char", stop_words=set("，？"))
    tokens = tokenizer(items, key=lambda x: x['stem'])
    ret = next(tokens)
    ans = ['文', '具', '店', '有', '$', '600', '$', '本', '练', '习', '本', '卖', '出', '一',
           '些', '后', '还', '剩', '$', '4', '$', '包', '每', '包', '$', '25', '$', '本', '卖', '出', '多', '少', '本']
    assert ret == ans


def test_Tokenizer():
    items = ['The stationery store has 600 exercise books, and after selling some,\
        there are still 4 packs left, 25 each, how many are sold?']
    ans = [
        'The', 'stationery', 'store', 'has', '600', 'exercise',
        'books', 'and', 'after', 'selling', 'some', 'there', 'are', 'still',
        '4', 'packs', 'left', '25', 'each', 'how', 'many', 'are', 'sold'
    ]
    for tok in ['nltk', 'spacy']:
        tokenizer = get_tokenizer("pure_text",
                                  text_params={"tokenizer": tok, "stopwords": set(",?")})
        tokens = tokenizer(items)
        ret = next(tokens)
        assert ret == ans


def test_TokenizerBPE():
    items = ['The stationery store has $600$ exercise books, and after selling some,\
        there are still $4$ packs left, $25$ each, how many are sold?']
    ans = [
        ['h', 'e', ' ', 'st', 'at', 'io', 'n', 'er', 'y', ' ', 'st', 'o', 're', ' ',
         'h', 'as', ' $', '6', '00', '$ ', 'e', 'x', 'er', 'ci', 's', 'e', ' b', 'o',
         'o', 'k', 's', ', ', 'an', 'd', ' a', 'ft', 'er', ' ', 's', 'e', 'l', 'l',
         'in', 'g', ' ', 's', 'ome', ', ', 't', 'h', 'e', 're', ' ', 'are', ' ',
         'st', 'i', 'l', 'l', ' $', '4', '$ ', 'p', 'a', 'c', 'k', 's', ' ', 'left',
         ', ', '$', '25', '$ ', 'e', 'a', 'c', 'h', ', ', 'h', 'ow', ' m', 'an', 'y',
         ' ', 'are', ' ', 's', 'o', 'l', 'd']
    ]
    data_path = path_append(abs_current_dir(__file__),
                            "../../static/test_data/standard_luna_data.json", to_str=True)
    tokenizer = get_tokenizer("pure_text", text_params={"tokenizer": 'bpe', "stopwords": set(",?"),
                              "bpe_trainfile": data_path})
    tokens = tokenizer(items)
    ret = next(tokens)
    assert ret == ans


def test_SpaceTokenizer():
    items = ['文具店有 $600$ 本练习本，卖出一些后，还剩 $4$ 包，每包 $25$ 本，卖出多少本？']
    tokenizer = get_tokenizer("space", stop_words=[])
    tokens = tokenizer(items)
    ret = next(tokens)
    ans = ['文具店有', '$600$', '本练习本，卖出一些后，还剩', '$4$', '包，每包', '$25$', '本，卖出多少本？']
    assert ret == ans


def test_AstformulaTokenizer():
    items = ['文具店有 $600$ 本练习本，卖出一些后，还剩 $4$ 包，每包 $25$ 本，卖出多少本？']
    tokenizer = get_tokenizer("ast_formula")
    tokens = tokenizer(items)
    ret = next(tokens)
    # ans = ['文具店', 'textord', 'textord', 'textord',
    # '练习本', '卖出', '剩', 'textord', '包', '每包', 'textord', 'textord', '卖出']
    ans = ['文具店', '6', '0', '0', '练习本', '卖出', '剩', '4', '包', '每包', '2', '5', '卖出']
    assert ret == ans


def test_PuretextTokenizer():
    items = ['文具店有 $600$ 本练习本，卖出一些后，还剩 $4$ 包，每包 $25$ 本，卖出多少本？']
    tokenizer = get_tokenizer("pure_text", stop_words=[])
    tokens = tokenizer(items)
    ret = next(tokens)
    ans = ['文具店', '600', '练习本', '卖出', '剩', '4', '包', '每包', '25', '卖出']
    assert ret == ans
    tokenizer = get_tokenizer("pure_text", stop_words=[], handle_figure_formula=None)
    tokens = tokenizer(items)
    ret = next(tokens)
    assert ret == ans
    tokenizer = get_tokenizer("pure_text", stop_words=[], handle_figure_formula='symbolize')
    tokens = tokenizer(items)
    ret = next(tokens)
    assert ret == ans
    with pytest.raises(ValueError):
        tokenizer = get_tokenizer("pure_text", stop_words=[], handle_figure_formula='wrong')


def test_CustomTokenizer():
    items = [{
        "stem": "文具店有 $600$ 本练习本，卖出一些后，还剩 $4$ 包，每包 $25$ 本，卖出多少本？",
        "options": ["1", "2"]
    }]
    tokenizer = get_tokenizer("custom", symbol='f')
    tokens = tokenizer(items, key=lambda x: x['stem'])
    ret = next(tokens)
    ans = ['文具店', '[FORMULA]', '练习本', '卖出', '剩', '[FORMULA]', '包', '每包', '[FORMULA]', '卖出']
    assert ret == ans
    items = [{
        "stem": "有公式$\\FormFigureID{1}$，如图$\\FigureID{088f15ea-xxx}$,若$x,y$满足约束条件公式$\\F\
            ormFigureBase64{2}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$",
        "options": ["1", "2"]
    }]
    tokenizer = get_tokenizer("custom", symbol='f', handle_figure_formula="symbolize")
    tokens = tokenizer(items, key=lambda x: x['stem'])
    ret = next(tokens)
    ret.pop(3)
    ans = ['公式', '[FORMULA]', '如图', '\\FigureID{088f15ea-xxx}', '[FORMULA]', '约束条件', '公式', '[FORMULA]',
           '\\SIFSep', '[FORMULA]', '最大值', '\\SIFBlank']
    ans.pop(3)
    assert ret == ans
