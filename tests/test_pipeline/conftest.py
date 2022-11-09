import pytest
from PIL import Image
from EduNLP.utils import abs_current_dir, path_append
from EduNLP.Vector import get_pretrained_model_info
from EduData import get_data


@pytest.fixture(scope="module")
def pretrained_elmo_for_property_prediction_dir():
    model_dir = path_append(abs_current_dir(__file__), "../../examples/test_model/elmo", to_str=True)
    url, _ = get_pretrained_model_info('elmo_pp_test')
    path = get_data(url, model_dir)
    return path


@pytest.fixture(scope="module")
def pretrained_elmo_for_knowledge_prediction_dir():
    model_dir = path_append(abs_current_dir(__file__), "../../examples/test_model/elmo", to_str=True)
    url, _ = get_pretrained_model_info('elmo_kp_test')
    path = get_data(url, model_dir)
    return path
