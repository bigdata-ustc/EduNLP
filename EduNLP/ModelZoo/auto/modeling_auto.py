from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from collections import OrderedDict

MODEL_MAPPING_NAMES = OrderedDict(
    [
        ("elmo", "ElmoModel"),
        ("bert", "BertModel")
    ]
)

CONFIG_MAPPING_NAMES = OrderedDict(
    [
        ("elmo", "PretrainedConfig"),
        ("bert", "PretrainedConfig")
    ]
)

TOKENIZER_MAPPING_NAMES = OrderedDict(
    [
        ("elmo", "ElmoTokenizer"),
        ("bert", "BertTokenizer")
    ]
)

MODEL_FOR_PROPERTY_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("elmo", "ElmoLMForPropertyPrediction"),
        ("bert", "BertForPropertyPrediction")
    ]
)

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        ("elmo", "ElmoLMForPreTraining"),
        ("bert", "BertForMaskedLM")
    ]
)

MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_PRETRAINING_MAPPING_NAMES
)

MODEL_FOR_PROPERTY_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_PROPERTY_PREDICTION_MAPPING_NAMES
)


class AutoModelForPretraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING


class AutoModelForPropertyPrediction(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PROPERTY_PREDICTION_MAPPING


AutoModelForPropertyPrediction = auto_class_update(AutoModelForPropertyPrediction, head_doc="property prediction")
AutoModelForPretraining = auto_class_update(AutoModelForPretraining, head_doc="pretraining")
