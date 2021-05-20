__AUTHOR__ = "Xin Wang"

from enum import IntFlag
import re


def cut(formula, preserve_braces=True, with_dollar=False,
        preserve_dollar=False, number_as_tag=False, preserve_src=True):  # pragma: no cover
    class States(IntFlag):
        CHAR = 0
        MATH = 1
        TAG = 2
        ESC = 3
        COMMAND = 4
        ARG = 5
        NUMBER = 8

    rv = []
    buffer = ''
    if with_dollar:
        state = States.CHAR
    else:
        state = States.MATH

    for c in formula:
        if state & States.NUMBER:
            if c.isdigit() or c == '.':
                buffer += c
                # c is consumed, continue
                continue
            else:
                state ^= States.NUMBER
                if number_as_tag:
                    if len(buffer) == 1:
                        rv.append(buffer)
                    elif '.' in buffer:
                        rv.append('{decimal}')
                    else:
                        rv.append('{integer}')
                else:
                    rv.append(buffer)
                buffer = ''

        if state == States.COMMAND:
            if buffer == '\\begin' or buffer == '\\end':
                state = States.ARG
                rv.append(buffer)
                buffer = c
                # c is consumed, continue
                continue
            elif c.isalpha():
                buffer += c
                # c is consumed, continue
                continue
            else:
                state = States.MATH
                if len(buffer) == 1:
                    buffer += c
                    rv.append(buffer)
                    buffer = ''
                    # c is consumed, continue
                    continue
                else:
                    rv.append(buffer)
                    buffer = ''

        if state == States.ESC:
            state = States.CHAR
            rv.append('\\' + c)
        elif state == States.CHAR:
            if c == '\\':
                state = States.ESC
            elif c.isdigit():
                state |= States.NUMBER
                buffer += c
            elif c == '{':
                state = States.TAG
                buffer += c
            elif c == '$':
                if preserve_dollar:
                    rv.append(c)
                state = States.MATH
            else:
                if c != ' ':
                    rv.append(c)
        elif state == States.TAG:
            if c == '}':
                state = States.CHAR
                buffer += c
                if not preserve_src:
                    if buffer.startswith('{img'):
                        buffer = '{img}'
                rv.append(buffer)
                buffer = ''
            else:
                buffer += c
        elif state == States.MATH:
            if c == '$':
                if preserve_dollar:
                    rv.append(c)
                state = States.CHAR
            elif c == '\\':
                state = States.COMMAND
                buffer += c
            elif c.isdigit():
                state |= States.NUMBER
                buffer += c
            else:
                if preserve_braces or (c != '{' and c != '}'):
                    if c != ' ':
                        rv.append(c)
        else:  # state == State.ARG
            if c == '}':
                state = States.MATH
                buffer += c
                rv.append(buffer)
                buffer = ''
            else:
                buffer += c
    if len(buffer) > 0:
        if state == States.NUMBER:
            if number_as_tag:
                if len(buffer) == 1:
                    rv.append(buffer)
                elif '.' in buffer:
                    rv.append('{decimal}')
                else:
                    rv.append('{integer}')
            else:
                rv.extend(buffer)
        else:
            rv.append(buffer)

    return rv


def reduce(fea):  # pragma: no cover
    rules = [
        ('a r c s i n', 'arcsin'),
        ('a r c c o s', 'arccos'),
        ('a r c t a n', 'arctan'),
        ('s i n h', 'sinh'),
        ('c o s h', 'cosh'),
        ('t a n h', 'tanh'),
        ('s i n', 'sin'),
        ('c o s', 'cos'),
        ('t a n', 'tan'),
        ('c o t', 'cot'),
        ('s e c', 'sec'),
        ('c s c', 'csc'),
        ('l g', 'lg'),
        ('l o g', 'log'),
        ('l n', 'ln'),
        ('m a x', 'max'),
        ('m i n', 'min'),
        ('{ i m g }', '{img}'),
        ('i m g', '{img}'),
        ('< u >', '{blank}'),
        ('  ', ' ')
    ]
    fea = ' '.join(fea)
    for a, b in rules:
        fea = fea.replace(a, b)
    return fea.strip().split()


def connect_char(words):  # pragma: no cover
    result = []
    buffer = ""
    for w in words:
        w = w.strip()
        if len(w) > 1:
            if len(buffer) > 0:
                result.append(buffer)
                buffer = ""
            result.append(w)

        elif len(w) == 1:
            if not w.isalpha():
                if len(buffer) > 0:
                    result.append(buffer)
                buffer = ""
                result.append(w)
            else:
                buffer += w
    if len(buffer) > 0:
        result.append(buffer)
        buffer = ""
    return result


def latex_parse(formula, preserve_braces=True, with_dollar=True,
                preserve_dollar=False, number_as_tag=False, preserve_src=True):  # pragma: no cover
    # cut
    formula_cut = cut(formula, preserve_braces, with_dollar,
                      preserve_dollar, number_as_tag, preserve_src)
    formula_reduce = reduce(formula_cut)
    formula_con = connect_char(formula_reduce)
    return formula_con


def linear_tokenize(formula, preserve_braces=True, number_as_tag=False, *args, **kwargs):
    """

    Parameters
    ----------
    formula
    preserve_braces
    number_as_tag
    args
    kwargs

    Returns
    -------

    Examples
    --------
    >>> linear_tokenize(r"{x + y}^\\frac{1}{2} + 1 = 0")
    ['{', 'x', '+', 'y', '}', '^', '\\\\frac', '{', '1', '}', '{', '2', '}', '+', '1', '=', '0']
    >>> linear_tokenize(r"ABC,AB,AC")
    ['ABC', ',', 'AB', ',', 'AC']
    """
    _formula_cut = cut(formula, preserve_braces=preserve_braces, number_as_tag=number_as_tag, *args, **kwargs)
    _formula_reduce = reduce(_formula_cut)
    _formula_con = connect_char(_formula_reduce)
    return _formula_con


# if __name__ == '__main__':
#     s = r"${x + y}^\frac{1}{2} + 1 = 0$"
#     l2 = re.split(r"(\$.+?\$)", s)
#
#     formula_cut = cut(s, with_dollar=True, preserve_braces=True)
#     formula_reduce = reduce(formula_cut)
#     formula_con = connect_char(formula_reduce)
#     print("s:", s)
#     print("formula_cut", formula_cut)
#     print("ormula_reduce", formula_reduce)
#     print("formula_con", formula_con)
