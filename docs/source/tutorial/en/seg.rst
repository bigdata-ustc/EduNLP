Component Segmentation
=========================

Educational resource is a kind of multimodal data, including data such as text, pictures, formulas and so on.
At the same time, it may also contain different components semantically, such as question stems, options, etc. Therefore, we first need to identify and segment the different components of educational resources:

* Semantic Component Segmentation
* Structural Component Segmentation

Main Processing Contents
---------------------------

1. Convert multiple-choice questions in the form of dict to qualified item by `Syntax parsing <parse.rst>`_;

2. The input items are segmented and grouped according to the element type.

Semantic Component Segmentation
---------------------------------

Because multiple-choice questions are given in the form of dict, it is necessary to convert them into text format while retaining their data relationship. This function can be realized by dict2str4sif function which can convert multiple-choice question items into character format and identify question stem and options。

Import Modules
+++++++++++++++++++++++

::

 from EduNLP.utils import dict2str4sif

Basic Usage
++++++++++++++++++

::

 >>> item = {
 ...     "stem": r"若复数$z=1+2 i+i^{3}$，则$|z|=$",
 ...     "options": ['0', '1', r'$\sqrt{2}$', '2'],
 ... }
 >>> dict2str4sif(item) # doctest: +ELLIPSIS
 '$\\SIFTag{stem_begin}$若复数$z=1+2 i+i^{3}$，则$|z|=$$\\SIFTag{stem_end}$$\\SIFTag{options_begin}$$\\SIFTag{list_0}$0$\\SIFTag{list_1}$1$\\SIFTag{list_2}$$\\sqrt{2}$$\\SIFTag{list_3}$2$\\SIFTag{options_end}$'

Optional additional parameters / interfaces
++++++++++++++++++++++++++++++++++++++++++++++++++

1.add_list_no_tag: if this parameter is true, it means that you need to count the labels in the options section.

::

 >>> dict2str4sif(item, add_list_no_tag=True) # doctest: +ELLIPSIS
 '$\\SIFTag{stem_begin}$若复数$z=1+2 i+i^{3}$，则$|z|=$$\\SIFTag{stem_end}$$\\SIFTag{options_begin}$$\\SIFTag{list_0}$0$\\SIFTag{list_1}$1$\\SIFTag{list_2}$$\\sqrt{2}$$\\SIFTag{list_3}$2$\\SIFTag{options_end}$'
 
 >>> dict2str4sif(item, add_list_no_tag=False) # doctest: +ELLIPSIS
 '$\\SIFTag{stem_begin}$若复数$z=1+2 i+i^{3}$，则$|z|=$$\\SIFTag{stem_end}$$\\SIFTag{options_begin}$0$\\SIFSep$1$\\SIFSep$$\\sqrt{2}$$\\SIFSep$2$\\SIFTag{options_end}$'

2.tag_mode: The location for the label can be selected using this parameter. 'delimiter' is to label both the beginning and the end,'head' is to label only the head, and 'tail' is to label only the tail.

::

 >>> dict2str4sif(item, tag_mode="head") # doctest: +ELLIPSIS
 '$\\SIFTag{stem}$若复数$z=1+2 i+i^{3}$，则$|z|=$$\\SIFTag{options}$$\\SIFTag{list_0}$0$\\SIFTag{list_1}$1$\\SIFTag{list_2}$$\\sqrt{2}$$\\SIFTag{list_3}$2'
 
 >>> dict2str4sif(item, tag_mode="tail") # doctest: +ELLIPSIS
 '若复数$z=1+2 i+i^{3}$，则$|z|=$$\\SIFTag{stem}$$\\SIFTag{list_0}$0$\\SIFTag{list_1}$1$\\SIFTag{list_2}$$\\sqrt{2}$$\\SIFTag{list_3}$2$\\SIFTag{options}$'

3.key_as_tag: If this parameter is false, this process will only adds $\SIFSep$ between the options without distinguishing the type of segmentation label.

::

 >>> dict2str4sif(item, key_as_tag=False)
 '若复数$z=1+2 i+i^{3}$，则$|z|=$0$\\SIFSep$1$\\SIFSep$$\\sqrt{2}$$\\SIFSep$2'

Structural Component Segmentation
------------------------------------------

This step is to segment sliced items. In this step, there is a depth option. You can select all positions or some labels for segmentation according to your needs, such as \SIFSep and \SIFTag. You can also select where to add labels, either at the head and tail or only at the head or tail.


There are two modes:

* linear mode: it is used for text processing (word segmentation using jieba library);

* ast mode: it is used to parse the formula.

Basic Segmentation process:

- Match components with regular expression matching

- Process the components with special structures, such as converting the base64 encoded picture to numpy form

- Classify the elements into each element group

- Enter the corresponding parameters as required to get the filtered results

Import Modules
+++++++++

::

 from EduNLP.SIF.segment import seg
 from EduNLP.SIF import sif4sci

Basic Usage
++++++++++++++++++

::

 >>> test_item = r"如图所示，则$\bigtriangleup ABC$的面积是$\SIFBlank$。$\FigureID{1}$"
 >>> seg(test_item)
 >>> ['如图所示，则', '\\bigtriangleup ABC', '的面积是', '\\SIFBlank', '。', \FigureID{1}]

Optional additional parameters/interfaces
++++++++++++++++++++++

1.describe: count the number of elements of different types

::

 >>> s.describe()
 {'t': 3, 'f': 1, 'g': 1, 'm': 1}

2.filter: this interface can screen out one or more types of elements.

Using this interface, you can pass in a "keep" parameter or a special character directly to choose what type of elements to retain.

Element type represented by symbol:

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

3.symbol: this interface can convert some types of data into special symbols

Element type represented by symbol:

-   "t": text
-   "f": formula
-   "g": figure
-   "m": question mark

::

 >>> seg(test_item, symbol="fgm")
 ['如图所示，则', '[FORMULA]', '的面积是', '[MARK]', '。', '[FIGURE]']
 >>> seg(test_item, symbol="tfgm")
 ['[TEXT]', '[FORMULA]', '[TEXT]', '[MARK]', '[TEXT]', '[FIGURE]']

In addition，sif4sci function is also provided, which can easily convert items into the result processed by Structural Component Segmentation

::

 >>> segments = sif4sci(item["stem"], figures=figures, tokenization=False)
 >>> segments
 ['如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形', 'ABC', '的斜边', 'BC', ', 直角边', 'AB', ', ', 'AC', '.', '\\bigtriangleup ABC', '的三边所围成的区域记为', 'I', ',黑色部分记为', 'II', ', 其余部分记为', 'III', '.在整个图形中随机取一点，此点取自', 'I,II,III', '的概率分别记为', 'p_1,p_2,p_3', ',则', '\\SIFChoice', \FigureID{1}]

- When calling this function, you can selectively output a certain type of data according to your needs

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

- Similar to seg function, sif4sci function also provides depth options to help with your research ----- By modifying the ``symbol`` parameter, different components can be transformed into specific markers.

::

 >>> sif4sci(item["stem"], figures=figures, tokenization=False, symbol="tfgm")
 ['[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[FORMULA]', '[TEXT]', '[MARK]', '[FIGURE]']
