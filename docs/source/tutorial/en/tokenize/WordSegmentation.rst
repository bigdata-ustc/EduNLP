Word segmentation
---------------------

Text-tokenization: A sentence (without formulas) consists of several "words" in order. The process of dividing a sentence into several words is called "Text-tokenization". According to the granularity of "words", it can be subdivided into "Word-tokenization" and "Char-tokenization".

::

   - Word-tokenization: each phrase is a token.

   - Char-tokenization: each character is a token.


Text-tokenization is divided into two main steps:

1. Text-tokenization:

   - Word-tokenization: use the word segmentation tool to segment and extract words from the question text. Our project supports `jieba`.

   - Char-tokenization: process text by character.

2. Filter: filter the specified stopwords.

   The default stopwords used in this project:`[stopwords] <https://github.com/bigdata-ustc/EduNLP/blob/master/EduNLP/meta_data/sif_stopwords.txt>`_
   You can also use your own stopwords. The following example demonstrates how to use.

Examples:

::
    
    >>> text = "三角函数是基本初等函数之一"
    >>> tokenize(text, granularity="word")
    ['三角函数', '初等', '函数']
    
    >>> tokenize(text, granularity="char")
    ['三', '角', '函', '数', '基', '初', '函', '数']
    
