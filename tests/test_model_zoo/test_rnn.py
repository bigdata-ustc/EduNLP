import torch
from EduNLP.ModelZoo.rnn import LM


idxs = torch.tensor([
    [1, 2, 3, 4, 0, 0],
    [1, 2, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 2, 0, 0, 0, 0]
])

lens = torch.tensor([4,2,1,2])

rnn = LM(rnn_type="lstm", vocab_size=20, embedding_dim=5, hidden_size=10)
output, hn = rnn(idxs, lens)

print("[output]", output)
print("[hn]", hn)