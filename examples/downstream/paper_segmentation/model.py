import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.modeling_outputs import ModelOutput
from EduNLP.ModelZoo.base_model import BaseModel
import numpy as np
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def zero_state(module, batch_size, device):
    # * 2 is for the two directions
    return torch.zeros(module.num_layers * 2, batch_size, module.hidden_dim).to(device), \
           torch.zeros(module.num_layers * 2, batch_size, module.hidden_dim).to(device)


def unsort(sort_order):
    result = [-1] * len(sort_order)
    for i, index in enumerate(sort_order):
        result[index] = i
    return result

class SentenceEncoder(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout=0,
                            bidirectional=True) # batch_first=False

    def forward(self, sentence_embs, sentence_lens):
        """
        Max-pooling for sentence representations 
        """
        batch_size = sentence_embs.shape[1]
        device = sentence_embs.device
        s = zero_state(self, batch_size, device=device)
        packed_tensor = pack_padded_sequence(sentence_embs, sentence_lens)
        packed_output, _ = self.lstm(packed_tensor, s)
        padded_output, lengths = pad_packed_sequence(packed_output)  # (max sentence len, batch, 256*2)
        maxes = torch.zeros(batch_size, padded_output.size(2)).to(device)
        for i in range(batch_size):
            maxes[i, :] = torch.max(padded_output[:lengths[i], i, :], 0)[0]

        return maxes

class TrainForPaperSegOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None


class PaperSegModel(BaseModel):
    def __init__(self, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sent_batch_size = None

        self.sentence_encoder = SentenceEncoder(self.embed_dim,
                                                self.hidden_dim,
                                                self.num_layers)
        self.lstm = nn.LSTM(input_size=self.hidden_dim*2,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout=0,
                            bidirectional=True) # batch_first=False
        self.full_connect = nn.Linear(self.hidden_dim*2, 2)  # 2 label
        # self.reset_parameters()

        self.criterion = torch.nn.CrossEntropyLoss()

        self.config = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        self.config['architecture'] = 'PaperSegModel'

    def set_sentence_batch(self, sent_batch_size):
        self.sent_batch_size = sent_batch_size

    def make_bach_sentences(self, sentences, lens, sent_batch_size):
        idx = 0
        batch_sentences, batch_lens = [], []
        while idx < len(sentences):
            next_idx = idx + sent_batch_size if idx + sent_batch_size <= len(sentences) else len(sentences)

            max_length = max( lens[idx: next_idx] )
            padded_sentences = [self.pad_sent(s, max_length) for s in sentences[idx: next_idx]]
            padded_sentences = torch.stack(padded_sentences, dim=1)

            batch_sentences.append( padded_sentences  )
            batch_lens.append( lens[idx: next_idx] )
            idx = next_idx
        return batch_sentences, batch_lens

    def reset_parameters(self):
        for name, param in self.lstm.named_parameters():
            if name.startswith('weight'):
                init.orthogonal_(param)
            else:
                assert name.startswith('bias')
                init.constant_(param, 0.)
        for name, param in self.full_connect.named_parameters():
            if name.startswith('weight'):
                init.orthogonal_(param)
            else:
                assert name.startswith('bias')
                init.constant_(param, 0.)

    def pad_sent(self, seq, max_length):
        s_length = seq.size()[0]
        padded = F.pad(seq, (0, 0, 0, max_length - s_length))  # (1, 1, max_length, 300)
        return padded
    
    def pad_document(self, d, max_document_length):
        d_length = d.size()[0]
        v = d.unsqueeze(0).unsqueeze(0)
        padded = F.pad(v, (0, 0, 0, max_document_length - d_length ))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)

    def forward(self, documents=None, tags=None):  # batch [documnts]\
        """
            documnts:
                [ sentences[word_embeddings[Tensor([300]), ... ], ...], ...]
            tags:
                torch.Tensor()
        """
        batch_size = len(documents)
        device = documents[0][0][0].device
        document_lens = []
        all_sentences = []
        for document in documents:
            all_sentences.extend(document)
            document_lens.append(len(document))

        # sentence 排序
        all_sentence_lens = [s.size()[0] for s in all_sentences]
        sort_order = np.argsort(all_sentence_lens)[::-1]
        sorted_sentences = [all_sentences[i] for i in sort_order]
        sorted_lengths = [s.size()[0] for s in sorted_sentences]

        # sentence 编码
        if self.sent_batch_size is not None:
            all_encoded_sentences = []
            all_sent_embs, all_sent_lens = self.make_bach_sentences(sorted_sentences, sorted_lengths, self.sent_batch_size)
            for batch_sent_embs, batch_sent_lens in zip(all_sent_embs, all_sent_lens):
                batch_encoded_sentences = self.sentence_encoder(batch_sent_embs, batch_sent_lens)
                all_encoded_sentences.extend(batch_encoded_sentences)
            all_encoded_sentences = torch.stack(all_encoded_sentences, dim=0)
        else:
            max_length = max(all_sentence_lens)
            sorted_padded_sentences = [self.pad_sent(s, max_length) for s in sorted_sentences]
            sorted_padded_sentences = torch.stack(sorted_padded_sentences, dim=1)
            all_encoded_sentences = self.sentence_encoder(sorted_padded_sentences, sorted_lengths)
        unsort_order = torch.LongTensor(unsort(sort_order)).to(device)
        unsorted_encodings = all_encoded_sentences.index_select(0, unsort_order)

        index = 0
        encoded_documents = []
        for sentences_count in document_lens:
            end_index = index + sentences_count
            encoded_documents.append(unsorted_encodings[index: end_index, :])
            index = end_index

        # document 排序
        max_doc_size = np.max(document_lens)
        ordered_document_idx = np.argsort(document_lens)[::-1]
        ordered_doc_sizes = sorted(document_lens)[::-1]
        ordered_documents = [encoded_documents[idx] for idx in ordered_document_idx]
        padded_docs = [self.pad_document(d, max_doc_size) for d in ordered_documents]
        docs_tensor = torch.cat(padded_docs, 1)
        packed_docs = pack_padded_sequence(docs_tensor, ordered_doc_sizes)
        sentence_lstm_output, _ = self.lstm(packed_docs, zero_state(self, batch_size=batch_size, device=device))
        padded_x, _ = pad_packed_sequence(sentence_lstm_output)

        doc_outputs = []
        for i, doc_len in enumerate(ordered_doc_sizes):
            # doc_outputs.append(padded_x[0:doc_len - 1, i, :])  # -1 to remove last prediction
            doc_outputs.append(padded_x[:, i, :])

        unsorted_doc_outputs = [doc_outputs[i] for i in unsort(ordered_document_idx)]
        self.sentence_outputs = torch.cat(unsorted_doc_outputs, 0)

        logits = self.full_connect(self.sentence_outputs)
        loss = None
        if tags is not None:
            loss = self.criterion(logits, tags.view(-1))

        return TrainForPaperSegOutput(
            loss=loss,
            logits=logits
        )

    @classmethod
    def from_config(cls, config_path, **kwargs):
        with open(config_path, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            model_config.update(kwargs)
            return cls(
                embed_dim=model_config["embed_dim"],
                hidden_dim=model_config["hidden_dim"],
                num_layers=model_config["num_layers"]
            )
    
    def save_config(self, config_dir):
        config_path = os.path.join(config_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as wf:
            json.dump(self.config, wf, ensure_ascii=False, indent=2)