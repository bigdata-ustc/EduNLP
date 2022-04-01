import torch
from ..meta import Vector
from EduNLP.ModelZoo.QuesNet import QuesNet
from EduNLP.Pretrain import Question


class QuesNetModel(Vector):
    def __init__(self, pretrained_dir, tokenizer, device="cpu"):
        """
        Parameters
        ----------
        pretrained_dir: str
            the dirname to pretrained model
        device: str
            cpu or cuda, default is cpu
        tokenizer: QuesNetTokenizer
            QuesNet  tokenzier
        """
        self.device = device
        self.model = QuesNet.from_pretrained(pretrained_dir, tokenizer)

    def __call__(self, item: Question):
        """

        Parameters
        ----------
        item : Question
            namedtuple, ['id', 'content', 'answer', 'false_options', 'labels']
        """
        _, v_embs, i_embs = self.model(self.model.make_batch(item, device=self.device))
        return i_embs[:, 2: -2, :], v_embs

    def infer_vector(self, item: Question) -> torch.Tensor:
        """ get question embedding with QuesNet

        Parameters
        ----------
        item : Question
            namedtuple, ['id', 'content', 'answer', 'false_options', 'labels']
        """
        return self.model(self.model.make_batch(item))[1]

    def infer_tokens(self, item: Question) -> torch.Tensor:
        """ get token embeddings with QuesNet

        Parameters
        ----------
        item : Question
            namedtuple, ['id', 'content', 'answer', 'false_options', 'labels']

        Returns
        -------
        torch.Tensor
            meta_emb + word_embs
        """
        return self.model(self.model.make_batch(item))[2][:, 2:-2, :]

    @property
    def vector_size(self):
        return self.model.feat_size
