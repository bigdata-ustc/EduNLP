import numpy as np
from pathlib import PurePath
from transformers import AutoModel
# from .const import UNK, PAD
# from .meta import Vector
import torch
from DisenQNet import DisenQNet



"""
注意修改DisenQNet中输入数据的格式，使用tokenizer处理后的令牌序列？
"""

class DisenQModel(object): # Vector
    """
    Examples
    
    """
    def __init__(self, pretrained_model, tokenizer=None):
        self.model = DisenQNet.load().disen_q_net
        if tokenizer:
            self.model.resize_token_embeddings(len(tokenizer.tokenizer))

    def __call__(self, items: dict):
        # 1, sent_len, embedding_size
        tokens = self.model(**items).last_hidden_state
        return tokens

    def infer_vector(self, items: dict) -> torch.Tensor:
        vector = self(items)
        return vector[:, 0, :]

    def infer_tokens(self, items: dict, return_special_tokens=False) -> torch.Tensor:
        tokens = self(items)
        if return_special_tokens:
            # include embedding of [CLS] and [SEP]
            return tokens
        else:
            # ignore embedding of [CLS] and [SEP]
            return tokens[:, 1:-1, :]

    @property
    def vector_size(self):
        return self.model.config.hidden_size
