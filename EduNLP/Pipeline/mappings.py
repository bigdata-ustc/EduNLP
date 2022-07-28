from collections import OrderedDict
from transformers.models.auto.auto_factory import _LazyAutoMapping

I2V_MAPPING_NAMES = OrderedDict(
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

TASK_MAPPING = {
    "property-prediction": MODEL_FOR_PROPERTY_PREDICTION_MAPPING_NAMES
}
