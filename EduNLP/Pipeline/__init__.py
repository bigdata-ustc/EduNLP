from .base import Pipeline, PreProcessingPipeline
from .mappings import TASK_MAPPING, TOKENIZER_MAPPING_NAMES
from .property_prediction import PropertyPredictionPipeline
from .knowledge_prediction import KnowledgePredictionPipeline
from ..Pretrain import PretrainedEduTokenizer
from ..ModelZoo.base_model import BaseModel
from ..Vector.t2v import get_pretrained_model_info
from ..constant import MODEL_DIR
from EduData import get_data
from typing import Optional, Union, List

__all__ = ["pipeline"]

SUPPORTED_TASKS = {
    "pre-process": {
        "impl": Pipeline,
        "default": None
    },
    "property-prediction": {
        "impl": PropertyPredictionPipeline,
        "default": "elmo_for_property_prediction_test_256"
    },
    "knowledge-prediction": {
        "impl": KnowledgePredictionPipeline,
        "default": "elmo_for_knowledge_prediction_test_256"
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

    Examples
    ----------
    >>> processor = pipeline(task="property-prediction") # doctest: +SKIP
    >>> item = "如图所示，则三角形ABC的面积是_。"
    >>> processor(item) # doctest: +SKIP
    """
    if preprocess is None and task is None and model is None:
        raise RuntimeError("Please specify at least the model to use or task to do!")
    elif model is None and tokenizer is not None:
        raise RuntimeError("Specified tokenizer but no model is not allowed!")
    elif task is None and model is not None:
        raise RuntimeError("Please specify the task.")
    elif task is None:
        task = "pre-process"

    if task == "pre-process":
        return PreProcessingPipeline(pipe_names=preprocess)

    if task in SUPPORTED_TASKS:
        targeted_task = SUPPORTED_TASKS[task]
    else:
        raise KeyError(f"Unknown task {task}")
    if pipeline_class is None:
        pipeline_class = targeted_task["impl"]
    if model is None or isinstance(model, str):
        # TODO: 1. waiting for ModelHub and TEST
        #       2. Check if the specified model and task are matched
        # pretrained_name = targeted_task["default"] if model is None else model
        # model_url, model_name, *args = get_pretrained_model_info(pretrained_name)
        # model_path = get_data(model_url, MODEL_DIR)
        # model = TASK_MAPPING[task][model_name].from_pretrained(model_path)
        # tokenizer = TOKENIZER_MAPPING_NAMES[model_name].from_pretrained(model_path)
        pass
    elif isinstance(model, BaseModel) and isinstance(tokenizer, PretrainedEduTokenizer):
        model, tokenizer = model, tokenizer
    elif model is not None and tokenizer is not None:
        raise KeyError(f"Unknown model and tokenizer: {model} and {tokenizer}")

    return pipeline_class(model=model, task=task, tokenizer=tokenizer, preproc_pipe_names=preprocess, **kwargs)
