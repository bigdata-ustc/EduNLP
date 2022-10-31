from collections import OrderedDict
from ..Pretrain import ElmoTokenizer, BertTokenizer, QuesNetTokenizer, DisenQTokenizer
from ..ModelZoo.rnn import ElmoLMForPropertyPrediction, ElmoLMForKnowledgePrediction
from ..ModelZoo.bert import BertForPropertyPrediction, BertForKnowledgePrediction

TOKENIZER_MAPPING_NAMES = OrderedDict(
    [
        ("elmo", ElmoTokenizer),
        ("bert", BertTokenizer),
        ("quesnet", QuesNetTokenizer),
        ("disenq", DisenQTokenizer)
    ]
)

MODEL_FOR_PROPERTY_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("elmo", ElmoLMForPropertyPrediction),
        ("bert", BertForPropertyPrediction),
    ]
)

MODEL_FOR_KNOWLEDGE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("elmo", ElmoLMForKnowledgePrediction),
        ("bert", BertForKnowledgePrediction)
    ]
)

TASK_MAPPING = {
    "property-prediction": MODEL_FOR_PROPERTY_PREDICTION_MAPPING_NAMES,
    "knowledge-prediction": MODEL_FOR_KNOWLEDGE_PREDICTION_MAPPING_NAMES
}
