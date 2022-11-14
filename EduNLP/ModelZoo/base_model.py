import torch.nn as nn
import json
import os
from pathlib import Path
import torch
from transformers import PretrainedConfig
# import logging
from ..utils import logger


class BaseModel(nn.Module):
    base_model_prefix = ''

    def __init__(self):
        super(BaseModel, self).__init__()
        self.config = PretrainedConfig()

    def forward(self, *input):
        raise NotImplementedError

    def save_pretrained(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'pytorch_model.bin')
        model_path = Path(model_path)
        torch.save(self.state_dict(), model_path.open('wb'))
        self.save_config(output_dir)

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *args, **kwargs):
        config_path = os.path.join(pretrained_model_path, "config.json")
        model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
        model = cls.from_config(config_path, *args, **kwargs)
        loaded_state_dict = torch.load(model_path)
        loaded_keys = loaded_state_dict.keys()
        expected_keys = model.state_dict().keys()

        prefix = cls.base_model_prefix

        if set(loaded_keys) == set(expected_keys):
            # same architecture
            model.load_state_dict(loaded_state_dict)
        else:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)

            new_loaded_state_dict = {}
            if expects_prefix_module and not has_prefix_module:
                # add prefix
                for key in loaded_keys:
                    new_loaded_state_dict['.'.join([prefix, key])] = loaded_state_dict[key]
            if has_prefix_module and not expects_prefix_module:
                # remove prefix
                for key in loaded_keys:
                    if key.startswith(prefix):
                        new_loaded_state_dict['.'.join(key.split('.')[1:])] = loaded_state_dict[key]
            if has_prefix_module and expects_prefix_module:
                # both have prefix, only load the base encoder
                for key in loaded_keys:
                    if key.startswith(prefix):
                        new_loaded_state_dict[key] = loaded_state_dict[key]
            loaded_state_dict = new_loaded_state_dict
            model.load_state_dict(loaded_state_dict, strict=False)
        loaded_keys = loaded_state_dict.keys()
        missing_keys = set(expected_keys) - set(loaded_keys)
        if len(missing_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        elif len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        return model

    def save_config(self, config_dir):
        config_path = os.path.join(config_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as wf:
            json.dump(self.config.to_dict(), wf, ensure_ascii=False, indent=2)

    @classmethod
    def from_config(cls, config_path, *args, **kwargs):
        raise NotImplementedError
