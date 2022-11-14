Get Started
===============

*  `Standard Item Format <sif.rst>`_

*  `Syntax Parsing <tokenize.rst>`_

*  `Component Segmentation <seg.rst>`_

*  `Tokenization <tokenization.rst>`_

*  `Pre-training <pretrain.rst>`_

*  `Vectorization <vectorization.rst>`_

*  `Pipeline <pipeline.rst>`_

Main process
---------------

.. figure:: ../../_static/pipeline.png

* `Component Segmentation <seg.rst>`_ :  Segment items in SIF format according to the types of items, so that elements in different types(text, formulas, pictures, etc.) can be tokenized respectively.

* `Syntax Parsing <tokenize.rst>`_ :  parsing different components in different ways, including formula parsing, text parsing, etc., serves the tokenization process later. 

* `Tokenization <tokenization.rst>`_: Further process the result of component segmentation and syntax parsing, and finally the multi-modal tokenization sequence of the item is obtained.  

* `Vectorization <vectorization.rst>`_: Fed the list of tokenized items into pre-training models, so as to get the corresponding vectors of items.

* **Downstream** Apply the obtained vectors to downstream tasks.

Examples
---------

To help you quickly understand the functions of this project, this section only shows the usages of common function interface. Intermediate function modules (such as parse, formula, segment, etc.) and more subdivided interface methods are not shown. For further study, please refer to relevant documents.

------------------------------------------------------------

.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: start_galler
    :glob:
    
    Tokenization  <../../build/blitz/sif/sif4sci.ipynb>

    Vectorization  <../../build/blitz/i2v/get_pretrained_i2v.ipynb>

    Pipeline <../../build/blitz/pipeline/pipeline.ipynb>
