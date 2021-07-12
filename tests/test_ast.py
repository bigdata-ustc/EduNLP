import pytest

from EduNLP.Formula.ast import str2ast


def test_ast():
    ast_str_list = []
    # normal examples
    ast_str_list.append(r"{x + y}^\frac{\pi}{2} + 1 = x")
    ast_str_list.append(r"\color{#0FF} x = y")
    ast_str_list.append(r"x^2 + 1 = y")
    ast_str_list.append(r"\verb!x^2!")
    ast_str_list.append(r"\utilde{AB}")
    ast_str_list.append(r"\mathrm{Ab0}")
    ast_str_list.append(r"{1,2,3}")
    ast_str_list.append(r"\huge AB")
    ast_str_list.append(r"\underline{AB}")
    ast_str_list.append(r"\sqrt{\smash[b]{y}}")
    ast_str_list.append(r"\hbox{AA BB}")
    ast_str_list.append(r"abc\llap{abcdefghi}")
    ast_str_list.append(r"\raisebox{3em}{hi}")
    ast_str_list.append(r"\textcolor{#228B22}{F=ma}")
    ast_str_list.append(r"\displaystyle\sum_{i=1}^n")
    ast_str_list.append(r"\def\foo{x^2} \foo + \foo")
    ast_str_list.append(r"thank \hphantom{xyz} you")
    ast_str_list.append(r"\mathchoice{D}{T}{S}{SS}")
    ast_str_list.append(r"\bigotimes")
    ast_str_list.append(r"{AB}_b^c")
    ast_str_list.append(r"\left\{\begin{array}{c}2 x+y-2 \leq 0 \\ x-y-1 \geq 0 \\ y+1 \geq 0\end{array}\right.")
    ast_str_list.append(r"\cancel{5}")

    # work only when katex is in 'display' mode :
    ast_str_list.append(r"\begin{matrix} a & b \\ c & d \end{matrix}")
    ast_str_list.append(r"\begin{pmatrix} a&b\\c&d \end{pmatrix}")
    ast_str_list.append(r"\begin{matrix}k个\\ \overbrace{(-1)^{k-1}k,\cdots,(-1)^{k-1}k}\end{matrix}")

    # work only when 'trust' katex html func:
    ast_str_list.append(r"\href{https://katex.org}{katex}")
    ast_str_list.append(r"\htmlStyle{color: red;}{x}")
    ast_str_list.append(r"\url{www.baidu.com}")
    ast_str_list.append(r"\htmlId{bar}{x}")
    ast_str_list.append(r"\htmlClass{foo}{x}")
    ast_str_list.append("\\includegraphics[height=0.8em, totalheight=0.9em, \
    width=0.9em, alt=KA logo]{https://katex.org/img/khan-academy.png}")

    for ast_str in ast_str_list:
        str2ast(ast_str)
