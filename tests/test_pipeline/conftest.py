import pytest
from PIL import Image
from EduNLP.utils import abs_current_dir, path_append


@pytest.fixture(scope="module")
def pretrained_elmo_for_property_prediction_dir():
    _dir = path_append(abs_current_dir(__file__), "../../examples/test_model/elmo/elmo_pp", to_str=True)
    return _dir


@pytest.fixture(scope="module")
def pretrained_elmo_for_knowledge_prediction_dir():
    _dir = path_append(abs_current_dir(__file__), "../../examples/test_model/elmo/elmo_kp", to_str=True)
    return _dir
