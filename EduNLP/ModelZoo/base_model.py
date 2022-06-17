import torch.nn as nn
import json
import os
from pathlib import Path
import torch
from transformers import PretrainedConfig


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        # self.config = {k: v for k, v in locals().items() if k != "self"}
        self.config = PretrainedConfig()

    def forward(self, *input):
        raise NotImplementedError

    def save_pretrained(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'pytorch_model.bin')
        model_path = Path(model_path)
        torch.save(self.state_dict(), model_path.open('wb'))
        # config_path = os.path.join(output_dir, "config.json")
        self.save_config(output_dir)

    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        raise NotImplementedError

    def save_config(self, config_dir):
        # with open(config_path, "w", encoding="utf-8") as wf:
        #     json.dump(self.config, wf, ensure_ascii=False, indent=2)
        self.config.save_pretrained(config_dir)

    @classmethod
    def from_config(cls, config_path):
        raise NotImplementedError
