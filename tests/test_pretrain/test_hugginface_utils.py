from EduNLP.Pretrain.hugginface_utils import TokenizerForHuggingface
import pytest
import os


# TODO
class TestPretrainUtils:
    def test_hf_tokenzier(self):
        tokenizer = TokenizerForHuggingface(tokenize_method=None)
        tokenizer = TokenizerForHuggingface(add_special_tokens=True)
        assert isinstance(tokenizer.vocab_size, int)
        item = 'This is a test.'
        res = tokenizer.decode(tokenizer.encode(item))
        right_ans = '[CLS] [UNK] is a test. [SEP]'
        assert res == right_ans, res
        with pytest.raises(OSError) or pytest.raises(ValueError):
            tokenizer = TokenizerForHuggingface.from_pretrained('wrong_path')
