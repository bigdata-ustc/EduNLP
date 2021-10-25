Standard Item Format
=======================

version: 0.2

For the convenience of follow-up research and use, we need a unified test question grammar standard.

Grammar Rules
----------------

1. Only Chinese characters, Chinese and English punctuation and line breaks are allowed in the question text.

2. Represent underlines of blanks and brackets of choices with ``\$\SIFBlank\$`` and ``\$\SIFChoice\$`` respectively.

3. We use ``$\FigureID{ uuid }$`` or Base64 to represent pictures. Especially, ``$\FormFigureID{ uuid }$`` is used to represent formulas pictures.

4. Text format description: we represent text in different styles with ``$\textf{item,CHAR_EN}$``. Currently, we have defined some styles: b-bold, i-italic, u-underline, w-wave, d-dotted, t-title. CHAR_EN Labels can be mixed and sorted alphabetically. An example: $\textf{EduNLP, b}$ looks **EduNLP**

5. Other mathematical symbols like English letters, Roman characters and numbers need to be expressed in latex format, that is, embedded in ``$$`` .

6. For the entry standard of molecular formula, please refer to `INCHI <https://zh.wikipedia.org/wiki/%E5%9B%BD%E9%99%85%E5%8C%96%E5%90%88%E7%89%A9%E6%A0%87%E8%AF%86>`_ for the time being.

7. Currently, there are no requirements for latex internal syntax.

::

  1. Item -> CHARACTER|EN_PUN_LIST|CH_PUN_LIST|FORMULA|QUES_MARK
  2. EN_PUN_LIST -> [',', '.', '?', '!', ':', ';', '\'', '\"', '(', ')', ' ','_','/','|','\\','<','>','[',']','-']
  3. CH_PUN_LIST -> ['，', '。', '！', '？', '：','；', '‘', '’', '“', '”', '（', '）', ' ', '、','《','》','—','．']
  4. FORMULA -> $latex formula$ | $\FormFigureID{UUID}$ | $\FormFigureBase64{BASE64}$
  5. FIGURE -> $\FigureID{UUID}$ | $\FigureBase64{BASE64}$
  6. UUID -> [a-zA-Z\-0-9]+
  7. CHARACTER -> CHAR_EN | CHAR_CH
  8. CHAR_EN -> [a-zA-Z]+
  9. CHAR_CH -> [\u4e00-\u9fa5]+
  10. DIGITAL -> [0-9]+
  11. QUES_MARK -> $\SIFBlank$ | $\SIFChoice$


Tips
+++++++++++++++

1. Reserved characters and escape characters.

2. Numbers.

3. Choices and blanks.

4. A single number or letter is also required to be between ``$$`` (automatic verification could already realize it).

5. Try to make sure Chinese is not included in the latex formula such as ``\text{CHAR_CH}``.

6. When importing data using MySQL database, an ``\`` is automatically ignored which needs to be further processed as ``\\``.

Examples
-----------------

Standard Format:

::

 1. 若$x,y$满足约束条件$\\left\\{\\begin{array}{c}2 x+y-2 \\leq 0 \\\\ x-y-1 \\geq 0 \\\\ y+1 \\geq 0\\end{array}\\right.$，则$z=x+7 y$的最大值$\\SIFUnderline$'
 
 2. 已知函数$f(x)=|3 x+1|-2|x|$画出$y=f(x)$的图像求不等式$f(x)>f(x+1)$的解集$\\PictureID{3bf2ddf4-8af1-11eb-b750-b46bfc50aa29}$$\\PictureID{59b8bd14-8af1-11eb-93a5-b46bfc50aa29}$$\\PictureID{63118b3a-8b75-11eb-a5c0-b46bfc50aa29}$$\\PictureID{6a006179-8b76-11eb-b386-b46bfc50aa29}$$\\PictureID{088f15eb-8b7c-11eb-a86f-b46bfc50aa29}$

Non-standard Format:

1. Letters, numbers and mathematical symbols are mixed:

    For example:
    
    ``完成下面的2x2列联表，``
    
    ``（单位：m3）``
    
    ``则输出的n=``
    
2. Some special mathematical symbols are not represented by the latex formula:

    For example:
    
    ``命题中真命题的序号是 ①``
    
    ``AB是⊙O的直径，AC是⊙O的切线，BC交⊙O于点E．若D为AC的中点``
    
3. There are unicode encoded characters in the text:

    For example:
    ``则$a$的取值范围是（\u3000\u3000）``

Functions for judging whether text is in SIF format and converting to SIF format
--------------------------------------------------------------------------------------------------

Call the Library
++++++++
::

    from EduNLP.SIF import is_sif, to_sif

is_sif
+++++++++++

::

    >>> text1 = '若$x,y$满足约束条件' 
    >>> text2 = '$\\left\\{\\begin{array}{c}2 x+y-2 \\leq 0 \\\\ x-y-1 \\geq 0 \\\\ y+1 \\geq 0\\end{array}\\right.$，' 
    >>> text3 = '则$z=x+7 y$的最大值$\\SIFUnderline$'
    >>> text4 = '某校一个课外学习小组为研究某作物的发芽率y和温度x（单位...'
    >>> is_sif(text1)
    True
    >>> is_sif(text2)
    True
    >>> is_sif(text3)
    True
    >>> is_sif(text4)
    False

to_sif
+++++++++++

::

    >>> text = '某校一个课外学习小组为研究某作物的发芽率y和温度x（单位...'
    >>> to_sif(text)
    '某校一个课外学习小组为研究某作物的发芽率$y$和温度$x$（单位...'


Change Log
----------------

2021-05-18

Changed

1. Originally, we use ``\$\SIFUnderline\$`` and ``\$\SIFBracket\$`` to represent underlines of blanks and brackets of choices. Now we represent them with ``\$\SIFBlank\$`` and ``\$\SIFChoice\$``.

2. Originally, we used ``$\PictureID{ uuid }$`` to represent pictures, but now we use ``$\FigureID{ uuid }$`` instead. Especially, ``$\FormFigureID{ uuid }$`` is used to represent formulas pictures.

2021-06-28 
  
Added:

1. There should not be line breaks between the notation ``$$``.

2. Add text format description.
