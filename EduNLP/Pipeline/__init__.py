from .base import TokenizerPipeline
from .property_prediction import PropertyPredictionPipeline
from .components import TOKENIZER_PIPES
from typing import Optional, Union, Dict, Any, Tuple
from ..Pretrain import PretrainedEduTokenizer
from ..ModelZoo.base_model import BaseModel
from ..ModelZoo.rnn import ElmoLMForPropertyPrediction
from ..Pretrain.elmo_vec import ElmoTokenizer

SUPPORTED_TASKS = {
    "property-prediction": {
        "impl": PropertyPredictionPipeline,
        "default": (ElmoLMForPropertyPrediction, ElmoTokenizer)
    }
}


def task_pipeline(
        task: str = None,
        model: Optional[Union[BaseModel, str]] = None,
        tokenizer: Optional[PretrainedEduTokenizer] = None,
        pipeline_class: Optional[Any] = None,
        **kwargs
):
    if task is None and model is None:
        raise RuntimeError("Please specify at least the model to use or task to do!")
    elif model is None and tokenizer is not None:
        raise RuntimeError("Specified tokenizer but no model is not allowed!")
    elif task is None and model is not None:
        raise RuntimeError("Specified model but no task is not allowed!")

    if task in SUPPORTED_TASKS:
        targeted_task = SUPPORTED_TASKS[task]
    else:
        raise KeyError(f"Unknown task {task}")
    if pipeline_class is None:
        pipeline_class = targeted_task["impl"]
    if model is None:
        # TODO: Default model and its config
        model, tokenizer = targeted_task["default"]
    elif isinstance(model, str):
        model = BaseModel.from_pretrained(model)
        tokenizer = PretrainedEduTokenizer.from_pretrained(model)

    return pipeline_class(model=model, task=task, tokenizer=tokenizer, **kwargs)
