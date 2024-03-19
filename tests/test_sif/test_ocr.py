# 2024/3/5 @ yuheng

import pytest
import json

from EduNLP.SIF.segment import seg
from EduNLP.SIF.parser.ocr import ocr_formula_figure, FormulaRecognitionError
from unittest.mock import patch


def test_ocr(figure0, figure1, figure0_base64, figure1_base64):
    seg(
        r"如图所示，则$\FormFigureID{0}$的面积是$\SIFBlank$。$\FigureID{1}$",
        figures={
            "0": figure0,
            "1": figure1
        },
        convert_image_to_latex=True
    )
    s = seg(
        r"如图所示，则$\FormFigureBase64{%s}$的面积是$\SIFBlank$。$\FigureBase64{%s}$" % (figure0_base64, figure1_base64),
        figures=True,
        convert_image_to_latex=True
    )
    with pytest.raises(TypeError):
        s.append("123")
    seg_test_text = seg(
        r"如图所示，有三组$\textf{机器人,bu}$在踢$\textf{足球,b}$",
        figures=True
    )
    assert seg_test_text.text_segments == ['如图所示，有三组机器人在踢足球']


def test_ocr_formula_figure_exceptions(figure0_base64):
    """Simulate a non-200 status code"""
    with patch('EduNLP.SIF.parser.ocr.requests.post') as mock_post:
        mock_post.return_value.status_code = 404
        with pytest.raises(FormulaRecognitionError) as exc_info:
            ocr_formula_figure(figure0_base64, is_base64=True)
        assert "HTTP error 404" in str(exc_info.value)

    """Simulate an invalid JSON response"""
    with patch('EduNLP.SIF.parser.ocr.requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b"invalid_json_response"
        with pytest.raises(FormulaRecognitionError) as exc_info:
            ocr_formula_figure(figure0_base64, is_base64=True)
        assert "Error processing response" in str(exc_info.value)

    """Simulate image not recognized as a formula"""
    with patch('EduNLP.SIF.parser.ocr.requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = json.dumps({
            "data": {
                'success': 1,
                'is_formula': 0,
                'detect_formula': 0
            }
        }).encode('utf-8')
        with pytest.raises(FormulaRecognitionError) as exc_info:
            ocr_formula_figure(figure0_base64, is_base64=True)
        assert "Image is not recognized as a formula" in str(exc_info.value)
