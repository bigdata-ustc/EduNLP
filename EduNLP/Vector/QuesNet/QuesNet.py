import torch
from ..meta import Vector
from EduNLP.ModelZoo.QuesNet import QuesNet
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
            QuesNet  tokenzier
        """
        self.device = device
        if tokenizer is None:
            tokenizer = QuesNetTokenizer.from_pretrained(pretrained_dir)
        self.model = QuesNet.from_pretrained(pretrained_dir, tokenizer)

    def infer_vector(self, items: Question) -> torch.Tensor:
        """ get question embedding with QuesNet

        Parameters
        ----------
        items : Question
            namedtuple, ['id', 'content', 'answer', 'false_options', 'labels']
        """
        return self.model(self.model.make_batch(items, device=self.device))[1]

    def infer_tokens(self, items: Question) -> torch.Tensor:
        """ get token embeddings with QuesNet

        Parameters
        ----------
        items : Question
            namedtuple, ['id', 'content', 'answer', 'false_options', 'labels']

        Returns
        -------
        torch.Tensor
            meta_emb + word_embs
        """
        return self.model(self.model.make_batch(items, device=self.device))[2][:, 2:-2, :]

    @property
    def vector_size(self):
        return self.model.feat_size
