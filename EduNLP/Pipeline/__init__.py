from .base import Pipeline
from .property_prediction import PropertyPredictionPipeline
from typing import Optional, Union, Dict, Any, Tuple
from ..Pretrain import PretrainedEduTokenizer
from ..ModelZoo.base_model import BaseModel
from ..ModelZoo.rnn import ElmoLMForPropertyPrediction
from ..Pretrain.elmo_vec import ElmoTokenizer

SUPPORTED_TASKS = {
    "property-prediction": {
        "impl": PropertyPredictionPipeline,
        "default": (ElmoLMForPropertyPrediction, ElmoTokenizer, "elmo_for_property_prediction_test_256")
    }
}


def pipeline(
        task: str = None,
        model: Optional[Union[BaseModel, str]] = None,
        tokenizer: Optional[PretrainedEduTokenizer] = None,
        pipeline_class: Optional[Pipeline] = None,
        **kwargs
):
    """
    Parameters
    ----------
    task: str, required
    model: BaseModel or str, optional
    tokenizer: PretrainedEduTokenizer, optional
    pipeline_class: Pipeline, optional

    """
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
        # TODO: Default model and its config.
        #  Specifically, automatically download default pretrained model&tokenizer when users give only `task` as input.
        model, tokenizer, pretrained_name = targeted_task["default"]
    elif isinstance(model, str):
        model = BaseModel.from_pretrained(model)
        tokenizer = PretrainedEduTokenizer.from_pretrained(model)
    elif isinstance(model, BaseModel) and isinstance(tokenizer, PretrainedEduTokenizer):
        model, tokenizer = model, tokenizer
    else:
        raise KeyError(f"Unknown model and tokenizer: {model} and {tokenizer}")

    return pipeline_class(model=model, task=task, tokenizer=tokenizer, **kwargs)
