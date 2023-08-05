EduNLP.Tokenizer
=====================================

.. automodule:: EduNLP.Tokenizer
   :members:
   :imported-members:

AstFormulaTokenizer参数定义
#######################################

::
    Parameters
        ----------
        symbol : str, optional
            Elements to symbolize before tokenization, by default "gmas"
        figures : _type_, optional
            Info for figures in items, by default None
        """
   
CharTokenizer参数定义
#######################################

::
    """Tokenize text char by char. eg. "题目内容" -> ["题",  "目",  "内", 容"]

        Parameters
        ----------
        stop_words : str, optional
            stop_words to skip, by default "default"
        """

CustomTokenizer参数定义
#######################################

::
    """Tokenize SIF items by customized configuration

        Parameters
        ----------
        symbol : str, optional
            Elements to symbolize before tokenization, by default "gmas"
        figures : _type_, optional
            Info for figures in items, by default None
        kwargs: addtional configuration for SIF items
            including text_params, formula_params, figure_params, more details could be found in `EduNLP.SIF.sif4sci`
        """

PureTextTokenizer参数定义
#######################################

::
    """
        Treat all elements in SIF item as prue text. Spectially, tokenize formulas as text.

        Parameters
        ----------
        handle_figure_formula : str, optional
            whether to skip or symbolize special formulas( $\\FormFigureID{…}$ and $\\FormFigureBase64{…}),
            by default skip

SpaceTokenizer参数定义
#######################################        

::
    """
    Tokenize text by space. eg. "题目 内容" -> ["题目", "内容"]

    Parameters
    ----------
    stop_words : str, optional
        stop_words to skip, by default "default"
    """

EduNLP.Tokenizer.get_tokenizer参数定义
####################################### 

::
    Parameters
    ----------
    name: str
        the name of tokenizer, e.g. text, pure_text.
    args:
        the parameters passed to tokenizer
    kwargs:
        the parameters passed to tokenizer
    Returns
    -------
    tokenizer: Tokenizer