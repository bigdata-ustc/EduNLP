from collections import OrderedDict
from transformers.models.auto.auto_factory import _LazyAutoMapping
from ..ModelZoo.auto.modeling_auto import (
    TOKENIZER_MAPPING_NAMES,
    MODEL_FOR_PROPERTY_PREDICTION_MAPPING_NAMES,
    MODEL_FOR_PRETRAINING_MAPPING_NAMES
)

# We currently has not defined specific configs for different models yet
# All models are mapped to base class `PretrainedConfig`

MODEL_FOR_PROPERTY_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_PROPERTY_PREDICTION_MAPPING_NAMES
)

TASK_MAPPING = {
    "property-prediction": MODEL_FOR_PROPERTY_PREDICTION_MAPPING_NAMES
}

