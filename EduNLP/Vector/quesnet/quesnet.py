import torch
from typing import Union
from EduNLP.ModelZoo.quesnet import QuesNet
from EduNLP.Pretrain import Question, QuesNetTokenizer


class QuesNetModel(object):
    def __init__(self, pretrained_dir, tokenizer=None, device="cpu"):
        """
        Parameters
        ----------
        pretrained_dir: str
            the dirname to pretrained model
        device: str
            cpu or cuda, default is cpu
        tokenizer: QuesNetTokenizer
            quesnet  tokenzier
        """
        self.device = torch.device(device)
        if tokenizer is None:
            tokenizer = QuesNetTokenizer.from_pretrained(pretrained_dir)
        self.model = QuesNet.from_pretrained(pretrained_dir, tokenizer).to(device)

    def infer_vector(self, items: Union[Question, list]) -> torch.Tensor:
        """ get question embedding with quesnet

        Parameters
        ----------
        items : (Question, list)
            namedtuple, ['id', 'content', 'answer', 'false_options', 'labels']
            or a list of Questions
        """
        inputs = [items] if isinstance(items, Question) else items
        vector = self.model(self.model.make_batch(inputs, device=self.device))[1]
        return vector

    def infer_tokens(self, items: Union[Question, list]) -> torch.Tensor:
        """ get token embeddings with quesnet

        Parameters
        ----------
        items : Question
            namedtuple, ['id', 'content', 'answer', 'false_options', 'labels']
            or a list of Questions
        Returns
        -------
        torch.Tensor
            meta_emb + word_embs
        """
        inputs = [items] if isinstance(items, Question) else items
        vector = self.model(self.model.make_batch(inputs, device=self.device))[2]
        return vector[:, 2:-2, :]

    @property
    def vector_size(self):
        return self.model.feat_size
