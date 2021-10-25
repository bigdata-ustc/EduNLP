Structural Component Segmentation
------------------------------------

This step is to segment sliced items. In this step, there is a depth option. You can select all positions or some labels for segmentation according to your needs, such as \SIFSep and \SIFTag. You can also select where to add labels, either at the head and tail or only at the head or tail.


There are two modes:

* linear mode: it is used for text processing (word segmentation using jieba library);

* ast mode: it is used to parse the formula.

Basic Usage
++++++++++++++++++

::

 >>> test_item = r"如图所示，则$\bigtriangleup ABC$的面积是$\SIFBlank$。$\FigureID{1}$"
 >>> seg(test_item)
 >>> ['如图所示，则', '\\bigtriangleup ABC', '的面积是', '\\SIFBlank', '。', \FigureID{1}]

Optional additional parameters/interfaces
+++++++++++++++++++++++++++++++++++++++++++++

1.describe: count the number of elements of different types

::

 >>> s.describe()
 {'t': 3, 'f': 1, 'g': 1, 'm': 1}

2.filter: this interface can screen out one or more types of elements.

Using this interface, you can pass in a "keep" parameter or a special character directly to choose what type of elements to retain.

Element type represented by symbol:
   "t": text
   "f": formula
   "g": figure
   "m": question mark
   "a": tag
   "s": sep tag

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
