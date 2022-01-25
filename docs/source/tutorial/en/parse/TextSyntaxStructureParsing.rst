Text syntax structure parsing
--------------------------------

This section is mainly realized by EduNLP.SIF.Parse module. Its main function is to extract letters and numbers in the text and convert them into standard format.

This module is mainly used as an *middle module* to parse the input text. In general, users do not call this module directly.

Introduction of Main Content
+++++++++++++++++++++++++++++++++++++

1. Judge the type of the incoming text in the following order

* is_chinese: its function is to match Chinese characters[\u4e00-\u9fa5].

* is_alphabet: its function is to match alphabets other than formulas. Only the alphabets between two Chinese characters will be corrected (wrapped with $$), and the rest of the cases are regarded as formulas that do not conform to latex syntax.

* is_number: its function is to match numbers other than formulas. Only the numbers between two Chinese characters will be corrected, and the rest of the cases are regarded as formulas that do not conform to latex syntax.

2. Match latex formula

* If Chinese characters appear in latex, print warning only once.

* Use _is_formula_legal function, check the completeness and analyzability of latex formula, and report an error for formulas that do not conform to latex syntax.

Input
>>>>>>>

Type: str

Content：question text

::

   >>> text1 = '生产某种零件的A工厂25名工人的日加工零件数_   _'
   >>> text2 = 'X的分布列为(   )'
   >>> text3 = '① AB是⊙O的直径，AC是⊙O的切线，BC交⊙O于点E．AC的中点为D'
   >>> text4 = '支持公式如$\\frac{y}{x}$，$\\SIFBlank$，$\\FigureID{1}$，不支持公式如$\\frac{ \\dddot y}{x}$'

Parsing
>>>>>>>>>>>>>>>>>>>>

::

   >>> text_parser1 = Parser(text1)
   >>> text_parser2 = Parser(text2)
   >>> text_parser3 = Parser(text3)
   >>> text_parser4 = Parser(text4)

Related parameters description(?)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

- Try to convert text to standard format

::

   >>> text_parser1.description_list()
   >>> print('text_parser1.text:',text_parser1.text)
   text_parser1.text: 生产某种零件的$A$工厂$25$名工人的日加工零件数$\SIFBlank$
   >>> text_parser2.description_list()
   >>> print('text_parser2.text:',text_parser2.text)
   text_parser2.text: $X$的分布列为$\SIFChoice$

- Determine if the text has syntax errors

::

   >>> text_parser3.description_list()
   >>> print('text_parser3.error_flag: ',text_parser3.error_flag)
   text_parser3.error_flag:  1
   >>> text_parser4.description_list()
   >>> print('text_parser4.fomula_illegal_flag: ',text_parser4.fomula_illegal_flag)
   text_parser4.fomula_illegal_flag:  1
