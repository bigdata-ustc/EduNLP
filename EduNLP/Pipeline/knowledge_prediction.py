import torch

from .base import Pipeline, GenericTensor
from typing import Dict, Optional, Union
from torch import sigmoid


class KnowledgePredictionPipeline(Pipeline):
    def __init__(self, **kwargs):
        super(KnowledgePredictionPipeline, self).__init__(**kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        tokenize_params, forward_params, postprocess_params = pipeline_parameters, {}, {}
        return tokenize_params, forward_params, postprocess_params

    def _tokenize(self, input_, **tokenize_parameters) -> Dict[str, GenericTensor]:
        return self.tokenizer(input_, **tokenize_parameters)

    def _forward(self, model_inputs, **forward_params):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs, **postprocess_params):
        if 'num_classes_list' not in dir(self.model) or 'num_total_classes' not in dir(self.model):
            raise ValueError('model is not for knowledge prediction: ', self.model)
        outputs = model_outputs["logits"][0]
        start_idx = 0
        knowledge_list = []
        for num_classes in self.model.num_classes_list:
            level_prediction = torch.argmax(outputs[start_idx:start_idx + num_classes]) + start_idx
            knowledge_list.append(level_prediction)
            start_idx += num_classes
        outputs = outputs.detach().numpy()
        dict_knowledge = {
            "knowledge_list": knowledge_list,
            "knowledge_scores": outputs.tolist(),
        }
        return dict_knowledge
