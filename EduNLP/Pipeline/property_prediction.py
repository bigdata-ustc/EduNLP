from transformers.pipelines.base import Pipeline, GenericTensor
from .mappings import MODEL_FOR_PROPERTY_PREDICTION_MAPPING
from typing import Dict


class PropertyPredictionPipeline(Pipeline):
    def __init__(self, **kwargs):
        super(PropertyPredictionPipeline, self).__init__(**kwargs)

        self.check_model_type(MODEL_FOR_PROPERTY_PREDICTION_MAPPING)

    def _sanitize_parameters(self, **tokenizer_kwargs):
        preprocess_params = tokenizer_kwargs
        return preprocess_params, {}, {}

    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        if isinstance(inputs, dict):
            return self.tokenizer(**inputs, **tokenizer_kwargs)
        else:
            return self.tokenizer(inputs, **tokenizer_kwargs)

    def _forward(self, model_inputs, **forward_params):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs, top_k=1):
        outputs = model_outputs["logits"][0]
        outputs = outputs.numpy()
        properties = sigmoid(outputs)
        dict_property = [
            {"property": p.item()} for p in properties
        ]
        return dict_property

    def __call__(self, *args, **kwargs):
        pass
