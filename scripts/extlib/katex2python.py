# coding: utf-8
# 2021/5/30 @ tongshiwei
from pathlib import PurePath
from fire import Fire
import requests
import js2py
import tempfile


def get_katex_from_url(version, tar):
    katex_version = version
    url = "https://cdn.jsdelivr.net/npm/katex@{}/dist/katex.js".format(katex_version)
    ret = requests.get(url, allow_redirects=True)
    assert ret.status_code == 200, ret.status_code
    content = requests.get(url).content
    tar.write(content)
    return url

'''
    Note:
        In that some formulas can not parse well by katex.py for some js2py errors, 
        we need to manually omit a few codes after ketex.py is built.
        eg 1. Array.fill() error :
            # var.get('res').put('cols', var.get('Array').create(var.get('numCols')).callprop('fill', Js({'type':Js('align'),'align':var.get('colAlign')})))
'''
def update_katex_py(src=None, tar="katex.py"):
    src = "katex.js" if src is None else src
    if PurePath(src).suffix == ".js":
        print("%s -> %s" % (src, tar))
        js2py.translate_file("katex.js", tar)
    else:
        with tempfile.NamedTemporaryFile() as tmp_tar:
            print("katex version: %s" % src)
            url = get_katex_from_url(src, tmp_tar)
            src = tmp_tar.name
            print("%s -> %s" % (url, tar))
            js2py.translate_file(src, tar)


if __name__ == '__main__':
    Fire(update_katex_py)
