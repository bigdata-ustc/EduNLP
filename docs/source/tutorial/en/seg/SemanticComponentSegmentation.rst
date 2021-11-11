Semantic Component Segmentation
------------------------------------

Because multiple-choice questions are given in the form of dict, it is necessary to convert them into text format while retaining their data relationship. This function can be realized by dict2str4sif function which can convert multiple-choice question items into character format and identify question stem and options。


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
++++++++++++++++++++++++++++++++++++++++++++++++

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