# coding: utf-8
# 2024/3/5 @ yuheng
import json
import requests
from EduNLP.utils import image2base64


class FormulaRecognitionError(Exception):
    """Exception raised when formula recognition fails."""
    def __init__(self, message="Formula recognition failed"):
        self.message = message
        super().__init__(self.message)


def ocr_formula_figure(image_PIL_or_base64, is_base64=False):
    """
    Recognizes mathematical formulas in an image and returns their LaTeX representation.

    Parameters
    ----------
    image_PIL_or_base64 : PngImageFile or str
        The PngImageFile if is_base64 is False, or the base64 encoded string of the image if is_base64 is True.
    is_base64 : bool, optional
        Indicates whether the image_PIL_or_base64 parameter is an PngImageFile or a base64 encoded string.

    Returns
    -------
    latex : str
        The LaTeX representation of the mathematical formula recognized in the image.
        Raises an exception if the image is not recognized as containing a mathematical formula.

    Raises
    ------
    FormulaRecognitionError
        If the HTTP request does not return a 200 status code,
        if there is an error processing the response,
        if the image is not recognized as a mathematical formula.

    Examples
    --------
    >>> import os
    >>> from PIL import Image
    >>> from EduNLP.utils import abs_current_dir, path_append
    >>> img_dir = os.path.abspath(path_append(abs_current_dir(__file__), "..", "..", "..", "asset", "_static"))
    >>> image_PIL = Image.open(path_append(img_dir, "item_ocr_formula.png", to_str=True))
    >>> print(ocr_formula_figure(image_PIL))
    f(x)=\\left (\\frac {1}{3}\\right )^{x}-\\sqrt {x}}
    >>> import os
    >>> from PIL import Image
    >>> from EduNLP.utils import abs_current_dir, path_append, image2base64
    >>> img_dir = os.path.abspath(path_append(abs_current_dir(__file__), "..", "..", "..", "asset", "_static"))
    >>> image_PIL = Image.open(path_append(img_dir, "item_ocr_formula.png", to_str=True))
    >>> image_base64 = image2base64(image_PIL)
    >>> print(ocr_formula_figure(image_base64, is_base64=True))
    f(x)=\\left (\\frac {1}{3}\\right )^{x}-\\sqrt {x}}

    Notes
    -----
    This function relies on an external service "https://formula-recognition-service-47-production.env.iai.bdaa.pro/v1",
    and the `requests` library to make HTTP requests. Make sure the required libraries are installed before use.
    """
    url = "https://formula-recognition-service-47-production.env.iai.bdaa.pro/v1"

    if is_base64:
        image = image_PIL_or_base64
    else:
        image = image2base64(image_PIL_or_base64)

    data = [{
        'qid': 0,
        'image': image
    }]

    resp = requests.post(url, data=json.dumps(data))

    if resp.status_code != 200:
        raise FormulaRecognitionError(f"HTTP error {resp.status_code}: {resp.text}")

    try:
        res = json.loads(resp.content)
    except Exception as e:
        raise FormulaRecognitionError(f"Error processing response: {e}")

    res = json.loads(resp.content)
    data = res['data']
    if data['success'] == 1 and data['is_formula'] == 1 and data['detect_formula'] == 1:
        latex = data['latex']
    else:
        latex = None
        raise FormulaRecognitionError("Image is not recognized as a formula")

    return latex


def ocr(src, is_base64=False, figure_instances: dict = None):
    """
    Recognizes mathematical formulas within figures from a given source,
    which can be either a base64 string or an identifier for a figure within a provided dictionary.

    Parameters
    ----------
    src : str
        The source from which the figure is to be recognized.
        It can be a base64 encoded string of the image if is_base64 is True,
        or an identifier for the figure if is_base64 is False.
    is_base64 : bool, optional
        Indicates whether the src parameter is a base64 encoded string or an identifier, by default False.
    figure_instances : dict, optional
        A dictionary mapping figure identifiers to their corresponding PngImageFile, by default None.
        This is only required and used if is_base64 is False.

    Returns
    -------
    forumla_figure_latex : str or None
        The LaTeX representation of the mathematical formula recognized within the figure.
        Returns None if no formula is recognized or
        if the figure_instances dictionary does not contain the specified figure identifier when is_base64 is False.

    Examples
    --------
    >>> import os
    >>> from PIL import Image
    >>> from EduNLP.utils import abs_current_dir, path_append
    >>> img_dir = os.path.abspath(path_append(abs_current_dir(__file__), "..", "..", "..", "asset", "_static"))
    >>> image_PIL = Image.open(path_append(img_dir, "item_ocr_formula.png", to_str=True))
    >>> figure_instances = {"1": image_PIL}
    >>> src_id = r"$\\FormFigureID{1}$"
    >>> print(ocr(src_id[1:-1], figure_instances=figure_instances))
    f(x)=\\left (\\frac {1}{3}\\right )^{x}-\\sqrt {x}}
    >>> import os
    >>> from PIL import Image
    >>> from EduNLP.utils import abs_current_dir, path_append, image2base64
    >>> img_dir = os.path.abspath(path_append(abs_current_dir(__file__), "..", "..", "..", "asset", "_static"))
    >>> image_PIL = Image.open(path_append(img_dir, "item_ocr_formula.png", to_str=True))
    >>> image_base64 = image2base64(image_PIL)
    >>> src_base64 = r"$\\FormFigureBase64{%s}$" % (image_base64)
    >>> print(ocr(src_base64[1:-1], is_base64=True, figure_instances=True))
    f(x)=\\left (\\frac {1}{3}\\right )^{x}-\\sqrt {x}}

    Notes
    -----
    This function relies on `ocr_formula_figure` for the actual OCR (Optical Character Recognition) process.
    Ensure that `ocr_formula_figure` is correctly implemented and can handle base64 encoded strings and PngImageFile.
    """
    forumla_figure_latex = None
    if is_base64:
        figure = src[len(r"\FormFigureBase64") + 1: -1]
        if figure_instances is not None:
            forumla_figure_latex = ocr_formula_figure(figure, is_base64)
    else:
        figure = src[len(r"\FormFigureID") + 1: -1]
        if figure_instances is not None:
            figure = figure_instances[figure]
            forumla_figure_latex = ocr_formula_figure(figure, is_base64)

    return forumla_figure_latex
