import torch
from torch import nn
import torch.nn.functional as F


class ElmoBilm(nn.Module):
    """
    Embeddings From Language Model
    Train this model and get presentations of pre-training

    # Parameters

    vocab_size: `int`, required
        The size of vocabulary
    emb_size: `int`, required
        The dimensionality of embeddings layer vectors.
    hidden_size: `int`, optional, (default = `4096`)
        The dimensionality of hidden layer vectors.
    dropout_rate: `float`, optional, (default = `0.5`)
        The rate of dropout to be applied in the LSTM network.
    num_layers: `int`, optional, (default = 2)
        The layer number of LSTM.
    batch_first: `bool`, optional, (default = True)
        Whether to use batch_first, which lead to different shape of input and output.
    """

    def __init__(
            self,
            vocab_size: int,
            emb_size: int = 512,
            hidden_size: int = 1024,
            dropout_rate: float = 0.5,
            num_layers: int = 2
    ) -> None:
        super(ElmoBilm, self).__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocabulary_size = vocab_size
        self.num_layers = num_layers

        self.Embedding_forward = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.emb_size,
                                              padding_idx=0)
        self.Embedding_backward = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.emb_size,
                                               padding_idx=0)
        # self.lstm_forwards = [nn.LSTM(input_size=self.emb_size, hidden_size=self.hidden_size,
        #                               num_layers=1, bias=True, batch_first=True, proj_size=self.emb_size) for i in
        #                       range(self.num_layers)]
        # self.lstm_backwards = [nn.LSTM(input_size=self.emb_size, hidden_size=self.hidden_size,
        #                                num_layers=1, bias=True, batch_first=True, proj_size=self.emb_size) for i in
        #                        range(self.num_layers)]
        self.lstm_forward_1 = nn.LSTM(input_size=self.emb_size, hidden_size=self.hidden_size,
                                      num_layers=1, bias=True, batch_first=True, proj_size=self.emb_size)
        self.lstm_forward_2 = nn.LSTM(input_size=self.emb_size, hidden_size=self.hidden_size,
                                      num_layers=1, bias=True, batch_first=True, proj_size=self.emb_size)
        self.lstm_backward_1 = nn.LSTM(input_size=self.emb_size, hidden_size=self.hidden_size,
                                       num_layers=1, bias=True, batch_first=True, proj_size=self.emb_size)
        self.lstm_backward_2 = nn.LSTM(input_size=self.emb_size, hidden_size=self.hidden_size,
                                       num_layers=1, bias=True, batch_first=True, proj_size=self.emb_size)

        self.pred_layer = nn.Linear(in_features=self.emb_size, out_features=self.vocabulary_size, bias=True)
        self.dropout = nn.Dropout(self.dropout_rate)

        initrange = 1.0 / self.emb_size
        nn.init.uniform_(self.Embedding_forward.weight.data, -initrange, initrange)
        nn.init.uniform_(self.Embedding_backward.weight.data, -initrange, initrange)
        nn.init.uniform_(self.pred_layer.weight.data, -initrange, initrange)

    def forward(self, input_seq: list):
        """
        # Parameters
        inputs : `list`, required
            A `list` of shape `(batch_size, sequence_length)`, which contains token indices.
            Notice: Here we use `batch-first`, be careful about the tensor dimensionality.

        # Returns
        pred_forward : `torch.Tensor`
            Predicted probabilities of each token in forward LM,
            of shape `(batch_size, sequence_length, vocabulary_size)`.
        pred_backward : `torch.Tensor`
            Predicted probabilities of each token in backward LM,
            of shape `(batch_size, sequence_length, vocabulary_size)`.
        forward_hiddens : `torch.Tensor`
            Hidden states in forward LM, including embedding layer,
            of shape `(1+num_layers, batch_size, sequence_length, emb_size)`.
        backward_hiddens : `torch.Tensor`
            Hidden states in backward LM, including embedding layer,
            of shape `(1+num_layers, batch_size, sequence_length, emb_size)`
        """
        if torch.is_tensor(input_seq):
            seq_indices = input_seq
        else:
            seq_indices = torch.tensor(input_seq)
        embeddings_forward = self.dropout(self.Embedding_forward(seq_indices))
        embeddings_backward = self.dropout(self.Embedding_backward(seq_indices))
        self.lstm_forward_1.flatten_parameters()
        self.lstm_forward_2.flatten_parameters()
        self.lstm_backward_1.flatten_parameters()
        self.lstm_backward_2.flatten_parameters()
        lstm_forward_output = embeddings_forward
        lstm_backward_output = torch.flip(embeddings_backward, [1])
        forward_hiddens = embeddings_forward.unsqueeze(0)
        backward_hiddens = embeddings_backward.unsqueeze(0)
        # embeddings has the same shape as lstm outputs have
        # for i in range(self.num_layers):
        #     lstm_forward_output, _ = self.lstm_forwards[i](lstm_forward_output)
        #     lstm_backward_output, _ = self.lstm_backwards[i](lstm_backward_output)
        #     forward_hiddens = torch.cat((forward_hiddens, lstm_forward_output.unsqueeze(0)), dim=0)
        #     backward_hiddens = torch.cat((backward_hiddens, lstm_backward_output.unsqueeze(0)), dim=0)
        #     lstm_forward_output = self.dropout(lstm_forward_output)
        #     lstm_backward_output = self.dropout(lstm_backward_output)
        lstm_forward_output, _ = self.lstm_forward_1(lstm_forward_output)
        lstm_backward_output, _ = self.lstm_backward_1(lstm_backward_output)
        forward_hiddens = torch.cat((forward_hiddens, lstm_forward_output.unsqueeze(0)), dim=0)
        backward_hiddens = torch.cat((backward_hiddens, lstm_backward_output.unsqueeze(0)), dim=0)
        lstm_forward_output = self.dropout(lstm_forward_output)
        lstm_backward_output = self.dropout(lstm_backward_output)
        lstm_forward_output, _ = self.lstm_forward_2(lstm_forward_output)
        lstm_backward_output, _ = self.lstm_backward_2(lstm_backward_output)
        forward_hiddens = torch.cat((forward_hiddens, lstm_forward_output.unsqueeze(0)), dim=0)
        backward_hiddens = torch.cat((backward_hiddens, lstm_backward_output.unsqueeze(0)), dim=0)
        lstm_forward_output = self.dropout(lstm_forward_output)
        lstm_backward_output = self.dropout(lstm_backward_output)

        pred_forward = F.softmax(input=self.pred_layer(lstm_forward_output), dim=-1)
        pred_backward = F.softmax(input=self.pred_layer(lstm_backward_output), dim=-1)

        return pred_forward, pred_backward, forward_hiddens, backward_hiddens

    def predict(self, input_seq: list):
        pred_forward, pred_backward, forward_hiddens, backward_hiddens = self.forward([input_seq])
        probabilities = pred_forward[0][-1]
        return probabilities.argmax().item()
