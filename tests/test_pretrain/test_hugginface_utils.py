from EduNLP.Pretrain.hugginface_utils import TokenizerForHuggingface
from transformers import AutoTokenizer
import os
os.environ["WANDB_DISABLED"] = "true"

# TODO
class TestPretrainUtils:
    def test_hf_tokenzier(self, pretrained_tokenizer_dir):
        tokenizer = TokenizerForHuggingface(tokenize_method=None)
        tokenizer = TokenizerForHuggingface(add_special_tokens=True)
        assert isinstance(tokenizer.vocab_size, int)
        item = 'This is a test.'
        res = tokenizer.decode(tokenizer.encode(item))
        right_ans = '[CLS] [UNK] is a test. [SEP]'
        assert res == right_ans, res
        items = ['This is a test.', 'This is a test 2.']
        res = tokenizer.decode(tokenizer.encode(items))
        right_ans = ['[CLS] [UNK] is a test. [SEP]', '[CLS] [UNK] is a test 2. [SEP]']
        assert res == right_ans, res

        tokenizer_hf = AutoTokenizer.from_pretrained("bert-base-chinese")
        tokenizer_hf.save_pretrained(pretrained_tokenizer_dir)

        tokenizer_hf = TokenizerForHuggingface.from_pretrained(pretrained_tokenizer_dir)
