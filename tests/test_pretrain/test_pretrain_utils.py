from EduNLP.Pretrain.pretrian_utils import EduVocab


# TODO
class TestPretrainUtils:
    def test_eduvocab(self):
        test = EduVocab()
        assert len(test) == 4
        token_list = ['An', 'apple', 'a', 'day', 'keeps', 'doctors', 'away']
        test.add_tokens(token_list)
        right_ans = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', 'An', 'apple', 'a', 'day', 'keeps', 'doctors', 'away']
        assert test.tokens == right_ans
        test_token_list = ['An', 'banana', 'is', 'a', 'kind', 'of', 'fruit']
        res = test.convert_sequence_to_token(test.convert_sequence_to_idx(test_token_list))
        right_ans = ['An', '[UNK]', '[UNK]', 'a', '[UNK]', '[UNK]', '[UNK]']
        assert res == right_ans
# t = TestPretrainUtils()
# t.test_eduvocab()

vocab = EduVocab()
print(vocab.tokens)
token_list = ['An', 'apple', 'a', 'day', 'keeps', 'doctors', 'away']
vocab.add_tokens(token_list)
test_token_list = ['An', 'banana', 'is', 'a', 'kind', 'of', 'fruit']
res = vocab.convert_sequence_to_token(vocab.convert_sequence_to_idx(test_token_list))
print(res)
