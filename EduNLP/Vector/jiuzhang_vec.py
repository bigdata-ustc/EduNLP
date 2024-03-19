from EduNLP.ModelZoo.jiuzhang import JiuzhangModel as Jiuzhang
from .meta import Vector
import torch


class JiuzhangModel(Vector):
    """
    Examples
    --------
    >>> from EduNLP.Pretrain import JiuzhangTokenizer
    >>> tokenizer = JiuzhangTokenizer("bert-base-chinese", add_special_tokens=False)
    >>> model = JiuzhangModel("bert-base-chinese")
    >>> item = ["有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$，若$x,y$满足约束",
    ... "有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$，若$x,y$满足约束"]
    >>> inputs = tokenizer(item, return_tensors='pt')
    >>> output = model(inputs)
    >>> output.shape
    torch.Size([2, 14, 768])
    >>> tokens = model.infer_tokens(inputs)
    >>> tokens.shape
    torch.Size([2, 12, 768])
    >>> tokens = model.infer_tokens(inputs, return_special_tokens=True)
    >>> tokens.shape
    torch.Size([2, 14, 768])
    >>> item = model.infer_vector(inputs)
    >>> item.shape
    torch.Size([2, 768])
    """

    def __init__(self, pretrained_dir, device="cpu"):
        self.device = device
        self.model = Jiuzhang.from_pretrained(pretrained_dir, ignore_mismatched_sizes=True).to(self.device)
        self.model.eval()

    def __call__(self, items: dict):
        self.cuda_tensor(items)
        tokens = self.model(**items).last_hidden_state
        return tokens

    def infer_vector(self, items: dict, pooling_strategy='CLS', **kwargs) -> torch.Tensor:
        vector = self(items)
        if pooling_strategy == 'CLS':
            return vector[:, 0, :]
        elif pooling_strategy == 'average':
            # the average of word embedding of the last layer
            # batch_size, sent_len, embedding_dim
            mask = items['attention_mask'].unsqueeze(-1).expand(vector.size())
            mul_mask = vector * mask
            # batch_size, embedding_dim
            return mul_mask.sum(1) / (mask.sum(1) + 1e-10)

    def infer_tokens(self, items: dict, return_special_tokens=False, **kwargs) -> torch.Tensor:
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
