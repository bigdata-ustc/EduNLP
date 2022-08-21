from .base import Pipeline, PreProcessingPipeline
from mappings import TASK_MAPPING
from .property_prediction import PropertyPredictionPipeline
from typing import Optional, Union, List
from ..Pretrain import PretrainedEduTokenizer
from ..ModelZoo.base_model import BaseModel

__all__ = ["pipeline"]

SUPPORTED_TASKS = {
    "pre-process": {
        "impl": Pipeline,
        "default": None
    },
    "property-prediction": {
        "impl": PropertyPredictionPipeline,
        "default": "elmo_for_property_prediction_test_256"
    }
}


def pipeline(
        task: str = None,
        model: Optional[Union[BaseModel, str]] = None,
        tokenizer: Optional[PretrainedEduTokenizer] = None,
        pipeline_class: Optional[Pipeline] = None,
        preprocess: Optional[List] = None,
        **kwargs
):
    """
    Parameters
    ----------
    task: str, required
    model: BaseModel or str, optional

    tokenizer: PretrainedEduTokenizer, optional

    pipeline_class: Pipeline, optional
        to specify Pipeline class
    preprocess: list, optional
        a list of names of pre-process pipes
    """
    if preprocess is None and task is None and model is None:
        raise RuntimeError("Please specify at least the model to use or task to do!")
    elif model is None and tokenizer is not None:
        raise RuntimeError("Specified tokenizer but no model is not allowed!")
    elif task is None and model is not None:
        raise RuntimeError("Specified model but no task is not allowed!")
    elif task is None:
        task = "pre-process"

    if task in SUPPORTED_TASKS:
        targeted_task = SUPPORTED_TASKS[task]
        model_mappings = TASK_MAPPING[task]
    else:
        raise KeyError(f"Unknown task {task}")
    if pipeline_class is None:
        pipeline_class = targeted_task["impl"]
    if model is None:
        # TODO: Default model and its config.
        #  Specifically, automatically download default pretrained model&tokenizer when users give only `task` as input.
        pretrained_name = targeted_task["default"]
    elif isinstance(model, str):
        # TODO: a mapping from name str to model class instance
        pass
    elif isinstance(model, BaseModel) and isinstance(tokenizer, PretrainedEduTokenizer):
        model, tokenizer = model, tokenizer
    elif model is not None and tokenizer is not None:
        raise KeyError(f"Unknown model and tokenizer: {model} and {tokenizer}")

    if task == "pre-process":
        return PreProcessingPipeline(pipe_names=preprocess)
    else:
        return pipeline_class(model=model, task=task, tokenizer=tokenizer, preproc_pipe_names=preprocess, **kwargs)
