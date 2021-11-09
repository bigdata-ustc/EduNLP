GensimSegTokenizer
=====================

By default, the pictures, separators, blanks in the question text and other parts of the incoming item are converted into special characters for data security and tokenization of text, formulas and labels. Also, the tokenizer uses linear analysis method for text and abstract analysis method of syntax tree for formulas.

Compared to GensimWordTokenizer, the main differences are:

* It provides the depth option for segmentation position, such as \SIFSep and \SIFTag.
* By default, labels are inserted in the header of item components (such as text and formula).