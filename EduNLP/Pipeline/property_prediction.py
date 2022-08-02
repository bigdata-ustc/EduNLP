from .base import Pipeline, GenericTensor
from typing import Dict, Optional, Union
from torch import sigmoid


class PropertyPredictionPipeline(Pipeline):
    def __init__(self, **kwargs):
        super(PropertyPredictionPipeline, self).__init__(**kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        tokenize_params, forward_params, postprocess_params = pipeline_parameters, {}, {}
        return tokenize_params, forward_params, postprocess_params

    def _tokenize(self, input_, **tokenize_parameters) -> Dict[str, GenericTensor]:
        return self.tokenizer(input_, **tokenize_parameters)

    def _forward(self, model_inputs, **forward_params):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs, **postprocess_params):
        outputs = model_outputs["logits"]
        outputs = outputs.numpy()
        dict_property = {"property": outputs.item()}
        return dict_property
