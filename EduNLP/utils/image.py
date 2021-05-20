# coding: utf-8
# 2021/5/20 @ tongshiwei

import base64
from io import BytesIO


def image2base64(img):
    buffered = BytesIO()
    img.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")
