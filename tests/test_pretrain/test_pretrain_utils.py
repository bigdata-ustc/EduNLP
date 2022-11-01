from EduNLP.Pretrain.pretrian_utils import EduVocab, PretrainedEduTokenizer
import pytest
import os


class TestPretrainUtils:
    def test_eduvocab(self):
        test = EduVocab(specials=['token1'])
        assert len(test) == 5
        token_list = ['An', 'apple', 'a', 'day', 'keeps', 'doctors', 'away']
        test.add_tokens(token_list)
        right_ans = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', 'token1',
                     'An', 'apple', 'a', 'day', 'keeps', 'doctors', 'away']
        assert test.tokens == right_ans
        assert test.vocab_size == len(right_ans)
        test_token_list = ['An', 'banana', 'is', 'a', 'kind', 'of', 'fruit']
        res = test.convert_sequence_to_token(test.convert_sequence_to_idx(test_token_list, bos=True, eos=True))
        right_ans = ['[BOS]', 'An', '[UNK]', '[UNK]', 'a', '[UNK]', '[UNK]', '[UNK]', '[EOS]']
        assert res == right_ans
        test.add_specials(['token2', 'token3'])
        right_ans = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', 'token1', 'token2', 'token3']
        test.special_tokens == right_ans

    def test_edu_tokenizer(self, pretrained_tokenizer_dir):
        test = EduVocab()
        token_list = ['An', 'apple', 'a', 'day', 'keeps', 'doctors', 'away']
        test.add_tokens(token_list)
        vocab_path = os.path.join(pretrained_tokenizer_dir, 'vocab.txt')
        test.save_vocab(vocab_path)

        test = PretrainedEduTokenizer(vocab_path=vocab_path, max_length=100)
        res = test('An apple a day keeps doctors away', padding='max_length')
        assert res['seq_idx'].shape[0] == 100
        res = test('An apple a day keeps doctors away', padding='longest')
        assert res['seq_idx'].shape[0] == res['seq_len']
        res = test('An apple a day keeps doctors away', padding='do_not_pad')
        assert res['seq_idx'].shape[0] == res['seq_len']
        with pytest.raises(ValueError):
            res = test('An apple a day keeps doctors away', padding='wrong_pad')

        res = test.decode(test.encode({'content': ['An', 'banana']}, key=lambda x: x['content']))
        right_ans = ['An', '[UNK]']
        print(res)
        assert res == right_ans, res


# t = TestPretrainUtils()
# t.test_eduvocab()


vocab = EduVocab()
print(vocab.tokens)
token_list = ['An', 'apple', 'a', 'day', 'keeps', 'doctors', 'away']
vocab.add_tokens(token_list)
test_token_list = ['An', 'banana', 'is', 'a', 'kind', 'of', 'fruit']
res = vocab.convert_sequence_to_token(vocab.convert_sequence_to_idx(test_token_list))
print(res)
