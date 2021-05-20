# coding: utf-8
# 2021/5/20 @ tongshiwei
import os
import pytest
from PIL import Image
from EduNLP.utils import abs_current_dir, path_append, image2base64


@pytest.fixture(scope="module")
def img_dir():
    return os.path.abspath(path_append(abs_current_dir(__file__), "..", "..", "asset", "_static"))


@pytest.fixture(scope="module")
def figure0(img_dir):
    return Image.open(path_append(img_dir, "item_formula.png", to_str=True))


@pytest.fixture(scope="module")
def figure1(img_dir):
    return Image.open(path_append(img_dir, "item_figure.png", to_str=True))


@pytest.fixture(scope="module")
def figure0_base64(figure0):
    return image2base64(figure0)


@pytest.fixture(scope="module")
def figure1_base64(figure1):
    return image2base64(figure1)
