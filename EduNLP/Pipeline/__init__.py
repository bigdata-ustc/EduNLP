from .base import TokenizerPipeline
from .property_prediction import PropertyPredictionPipeline
from .components import TOKENIZER_PIPES
from typing import Optional, Union, Dict, Any, Tuple
from ..Pretrain import PretrainedEduTokenizer
from ..ModelZoo.base_model import BaseModel
from ..ModelZoo.auto import AutoModelForPropertyPrediction, AutoConfig

SUPPORTED_TASKS = {
    "property-prediction": {
        "impl": PropertyPredictionPipeline,
        "model": AutoModelForPropertyPrediction
    }
}


# TODO:
# 1. AutoModel or something else, for model loading in Pipeline
# 2. AutoConfig or something else,
#    however, we use PretrainedConfig for all model structure now, whether or not to inherit and explicitly separate them
#    is TO BE DETERMINED.
# 3. AutoTokenizer or something else, for inferring Tokenizer from model of config name.
# 4. Some HuggingFace API usages haven't done completeness verification, generally in ModelZoo.auto module

def task_pipeline(
        task: str = None,
        model: Optional[BaseModel] = None,
        config: Optional[Union[str, PretrainedConfig]] = None,
        tokenizer: Optional[Union[str, PretrainedEduTokenizer]] = None,
        model_kwargs: Dict[str, Any] = None,
        pipeline_class: Optional[Any] = None,
        **kwargs
):
    if model_kwargs is None:
        model_kwargs = {}
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
        model = targeted_task["model"]

    if isinstance(config, str):
        config = AutoConfig.from_pretrained(config, _from_pipeline=task, **model_kwargs)
    elif config is None and isinstance(model, str):
        config = AutoConfig.from_pretrained(model, _from_pipeline=task, **model_kwargs)
    model = model.from_pretrained(model, config=config, task=task)

    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer
    return pipeline_class(model=model, task=task, **kwargs)
