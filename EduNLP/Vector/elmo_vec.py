import torch
from EduNLP.ModelZoo.rnn import ElmoLM
from .meta import Vector


class ElmoModel(Vector):
    def __init__(self, pretrained_dir: str, device="cpu"):
        """
        Parameters
        ----------
        pretrained_model_path: str
        """
        super(ElmoModel, self).__init__()
        self.device = device
        self.model = ElmoLM.from_pretrained(pretrained_dir).to(device)
        self.model.eval()

    def __call__(self, items: dict):
        self.cuda_tensor(items)
        outputs = self.model(**items)
        return outputs

    def infer_vector(self, items: dict, **kwargs) -> torch.Tensor:
        """
        get sentence vector embedding with ElmoModel
        Parameters
        ----------
        items: dict, {'seq_idx': tensor(),'seq_len':tensor()}, the tokens about question after tokenizer processing

        Returns:
            torch.Tensor: sentence embedding
        """
        outputs = self(items)
        item_embeds = torch.cat(
            (outputs.forward_output[torch.arange(len(items["seq_len"])), torch.tensor(items["seq_len"]) - 1],
             outputs.backward_output[torch.arange(len(items["seq_len"])), 0]),
            dim=-1)
        return item_embeds

    def infer_tokens(self, items, **kwargs) -> torch.Tensor:
        """
        get tokens embedding with ElmoModel
        Parameters
        ----------
        items: dict, {'seq_idx': tensor()}, the tokens about question after tokenizer processing

        Returns:
            torch.Tensor: token embedding
        """
        outputs = self(items)
        forward_hiddens = outputs.forward_output
        backward_hiddens = outputs.backward_output
        return torch.cat((forward_hiddens, backward_hiddens), dim=-1)

    @property
    def vector_size(self):
        return 2 * self.model.hidden_size
