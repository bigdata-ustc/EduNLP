成分分解
=========

由于教育资源是一种多模态数据，包含了诸如文本、图片、公式等数据结构；
同时在语义上也可能包含不同组成部分，例如题干、选项等，因此我们首先需要对教育资源的不同组成成分进行识别并进行分解：

* 语义成分分解
* 结构成分分解

主要处理内容
--------------------

1.将字典输入形式的选择题通过 `语法解析 <parse.rst>`_ 转换为符合条件的item；

2.将输入的item按照元素类型进行切分、分组。

语义成分分解
------------

由于选择题是以字典的形式给出，故需要将其在保留数据类型关系的情况下转换为文本格式。dict2str4sif函数就是实现此功能的一个模块，该模块可以将选择题形式的item转换为字符格式，并将题干和选项、各选项之间分割开来。

导入库
+++++++++

::

 from EduNLP.utils import dict2str4sif

基础使用方法
++++++++++++++++++

::

 >>> item = {
 ...     "stem": r"若复数$z=1+2 i+i^{3}$，则$|z|=$",
 ...     "options": ['0', '1', r'$\sqrt{2}$', '2'],
 ... }
 >>> dict2str4sif(item) # doctest: +ELLIPSIS
 '$\\SIFTag{stem_begin}$若复数$z=1+2 i+i^{3}$，则$|z|=$$\\SIFTag{stem_end}$$\\SIFTag{options_begin}$$\\SIFTag{list_0}$0$\\SIFTag{list_1}$1$\\SIFTag{list_2}$$\\sqrt{2}$$\\SIFTag{list_3}$2$\\SIFTag{options_end}$'

可选的的额外参数/接口
++++++++++++++++++++++

1.add_list_no_tag：当此参数为True较False时区别在于是否需要将选项部分的标签计数

::

 >>> dict2str4sif(item, add_list_no_tag=True) # doctest: +ELLIPSIS
 '$\\SIFTag{stem_begin}$若复数$z=1+2 i+i^{3}$，则$|z|=$$\\SIFTag{stem_end}$$\\SIFTag{options_begin}$$\\SIFTag{list_0}$0$\\SIFTag{list_1}$1$\\SIFTag{list_2}$$\\sqrt{2}$$\\SIFTag{list_3}$2$\\SIFTag{options_end}$'
 
 >>> dict2str4sif(item, add_list_no_tag=False) # doctest: +ELLIPSIS
 '$\\SIFTag{stem_begin}$若复数$z=1+2 i+i^{3}$，则$|z|=$$\\SIFTag{stem_end}$$\\SIFTag{options_begin}$0$\\SIFSep$1$\\SIFSep$$\\sqrt{2}$$\\SIFSep$2$\\SIFTag{options_end}$'

2.tag_mode:此参数为选择标签所在位置，delimiter为头尾都加标签，head为仅头部加标签，tail为仅尾部加标签

::

 >>> dict2str4sif(item, tag_mode="head") # doctest: +ELLIPSIS
 '$\\SIFTag{stem}$若复数$z=1+2 i+i^{3}$，则$|z|=$$\\SIFTag{options}$$\\SIFTag{list_0}$0$\\SIFTag{list_1}$1$\\SIFTag{list_2}$$\\sqrt{2}$$\\SIFTag{list_3}$2'
 
 >>> dict2str4sif(item, tag_mode="tail") # doctest: +ELLIPSIS
 '若复数$z=1+2 i+i^{3}$，则$|z|=$$\\SIFTag{stem}$$\\SIFTag{list_0}$0$\\SIFTag{list_1}$1$\\SIFTag{list_2}$$\\sqrt{2}$$\\SIFTag{list_3}$2$\\SIFTag{options}$'

3.key_as_tag:当其为False时则不区分切分标签的类型，而是仅在选项之间加入$\SIFSep$

::

 >>> dict2str4sif(item, key_as_tag=False)
 '若复数$z=1+2 i+i^{3}$，则$|z|=$0$\\SIFSep$1$\\SIFSep$$\\sqrt{2}$$\\SIFSep$2'

结构成分分解
------------

对切片后的item中的各个元素进行分词，提供深度选项，可以按照需求选择所有地方切分或者在部分标签处切分（比如\SIFSep、\SIFTag处）；对标签添加的位置也可以进行选择，可以在头尾处添加或仅在头或尾处添加。

具有两种模式:

* linear模式，用于对文本进行处理（使用jieba库进行分词）；

* ast模式，用于对公式进行解析。

基础分解流程：

- 使用正则匹配方法匹配出各个组成成分

- 对特殊结构的成分进行处理，如将base64编码的图片转为numpy形式

- 将当前元素分类放入各个元素组中

- 按照需求输入相应的参数得到筛选后的结果

导入库
+++++++++

::

 from EduNLP.SIF.segment import seg
 from EduNLP.SIF import sif4sci

基础使用方法
++++++++++++++++++

::

 >>> test_item = r"如图所示，则$\bigtriangleup ABC$的面积是$\SIFBlank$。$\FigureID{1}$"
 >>> seg(test_item)
 >>> ['如图所示，则', '\\bigtriangleup ABC', '的面积是', '\\SIFBlank', '。', \FigureID{1}]

可选的的额外参数/接口
++++++++++++++++++++++

1.describe：可以统计出各种类型元素的数量

::

 >>> s.describe()
 {'t': 3, 'f': 1, 'g': 1, 'm': 1}

2.filter：可以选择性的筛除某种或几种类型的元素

此接口可传入keep参数来选择需要保留的元素类型，也可直接传入特殊字符来筛除特定元素类型

各字母所代表的元素类型：

-   "t": text
-   "f": formula
-   "g": figure
-   "m": question mark
-   "a": tag
-   "s": sep tag

::

 >>> with s.filter("f"):
 ...     s
 ['如图所示，则', '的面积是', '\\SIFBlank', '。', \FigureID{1}]
 >>> with s.filter(keep="t"):
 ...     s
 ['如图所示，则', '的面积是', '。']

3.symbol:选择性的将部分类型的数据转换为特殊符号遮掩起来

symbol所代表的元素类型：

-   "t": text
-   "f": formula
-   "g": figure
-   "m": question mark

::

 >>> seg(test_item, symbol="fgm")
 ['如图所示，则', '[FORMULA]', '的面积是', '[MARK]', '。', '[FIGURE]']
 >>> seg(test_item, symbol="tfgm")
 ['[TEXT]', '[FORMULA]', '[TEXT]', '[MARK]', '[TEXT]', '[FIGURE]']

此外，当前还提供了sif4sci函数，其可以很方便的将item转换为结构成分分解后的结果

::

 >>> segments = sif4sci(item["stem"], figures=figures, tokenization=False)
 >>> segments
 ['如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形', 'ABC', '的斜边', 'BC', ', 直角边', 'AB', ', ', 'AC', '.', '\\bigtriangleup ABC', '的三边所围成的区域记为', 'I', ',黑色部分记为', 'II', ', 其余部分记为', 'III', '.在整个图形中随机取一点，此点取自', 'I,II,III', '的概率分别记为', 'p_1,p_2,p_3', ',则', '\\SIFChoice', \FigureID{1}]

- 调用此函数时，可以按照需求选择性的输出某一类型的数据

::

 >>> segments.formula_segments
 ['ABC',
 'BC',
 'AB',
 'AC',
 '\\bigtriangleup ABC',
 'I',
 'II',
 'III',
 'I,II,III',
 'p_1,p_2,p_3']

- 与seg函数类似，sif4sci也提供了标记化切分选项通过修改 ``symbol`` 参数来将不同的成分转化成特定标记，方便您的研究

::

 >>> sif4sci(item["stem"], figures=figures, tokenization=False, symbol="tfgm")
 ['[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[MARK]', '[FIGURE]']
