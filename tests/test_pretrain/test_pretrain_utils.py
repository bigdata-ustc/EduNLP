from EduNLP.Pretrain.pretrian_utils import EduVocab, PretrainedEduTokenizer, EduDataset
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
        test = EduVocab(corpus_items=[token_list])

    def test_edu_tokenizer(self, pretrained_tokenizer_dir):
        test = EduVocab()
        token_list = ['An', 'apple', 'a', 'day', 'keeps', 'doctors', 'away']
        test.add_tokens(token_list)
        vocab_path = os.path.join(pretrained_tokenizer_dir, 'vocab.txt')
        test.save_vocab(vocab_path)
        test = EduVocab(vocab_path=vocab_path)

        text = 'An apple a day keeps doctors away'
        tokenizer = PretrainedEduTokenizer(vocab_path=vocab_path, max_length=100)
        res = tokenizer(text, padding='max_length')
        assert res['seq_idx'].shape[0] == 100
        res = tokenizer(text, padding='longest')
        assert res['seq_idx'].shape[0] == res['seq_len']
        res = tokenizer(text, padding='do_not_pad')
        assert res['seq_idx'].shape[0] == res['seq_len']
        with pytest.raises(ValueError):
            res = tokenizer(text, padding='wrong_pad')
        tokenizer.add_tokens("[token]")
        tokenizer.add_specials("[special]")
        res = tokenizer.decode(tokenizer.encode({'content': ['An', 'banana']}, key=lambda x: x['content']))
        right_ans = ['An', '[UNK]']
        print(res)
        assert res == right_ans, res

        res = tokenizer.decode(tokenizer.encode([token_list]))
        assert res == [token_list]
        tokenizer.save_pretrained(f"{pretrained_tokenizer_dir}/save_dir")

    def test_edu_dateset(self, standard_luna_data, pretrained_tokenizer_dir):
        tokenizer = PretrainedEduTokenizer()
        tokenizer.set_vocab(standard_luna_data, key=lambda x: x["ques_content"])
        dataset = EduDataset(tokenizer,
                             items=standard_luna_data,
                             stem_key="ques_content")
        assert "seq_idx" in dataset[0].keys() and "seq_len" in dataset[0].keys()
        dataset.to_disk(f"{pretrained_tokenizer_dir}/dataset")

        local_dataset = EduDataset(tokenizer, f"{pretrained_tokenizer_dir}/dataset")
        assert local_dataset[0] == dataset[0]
