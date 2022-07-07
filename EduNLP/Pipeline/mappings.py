from collections import OrderedDict
from transformers.models.auto.auto_factory import _LazyAutoMapping

MODEL_MAPPING_NAMES = OrderedDict(
    [
        ("elmo", "ElmoModel"),
        ("bert", "BertModel")
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


TASK_MAPPING = {
    "property-prediction": MODEL_FOR_PROPERTY_PREDICTION_MAPPING_NAMES
}
