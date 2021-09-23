from EduNLP import logger
import multiprocessing
import transformers
from EduNLP.Tokenizer import PureTextTokenizer
from copy import deepcopy
import itertools as it
from EduNLP.SIF.sif import sif4sci
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from EduNLP.SIF import Symbol, FORMULA_SYMBOL, FIGURE_SYMBOL, QUES_MARK_SYMBOL, TAG_SYMBOL, SEP_SYMBOL


__all__ = ["BertTokenizer", "finetune_bert"]


class BertTokenizer(object):
    """

    Parameters
    ----------
    pretrain_model:
        used pretrained model

    Returns
    ----------

    Examples
    ----------
    >>> tokenizer = BertTokenizer()
    >>> item = "有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"
    >>> token_item = tokenizer(item)
    >>> print(token_item.input_ids[:10])
    [101, 1062, 2466, 1963, 1745, 21129, 166, 117, 167, 5276]
    >>> print(tokenizer.tokenize(item)[:10])
    ['公', '式', '如', '图', '[FIGURE]', 'x', ',', 'y', '约', '束']
    """
    def __init__(self, pretrain_model="bert-base-chinese"):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
        customize_tokens = []
        for i in [FORMULA_SYMBOL, FIGURE_SYMBOL, QUES_MARK_SYMBOL, TAG_SYMBOL, SEP_SYMBOL]:
            if i not in self.tokenizer.additional_special_tokens:
                customize_tokens.append(Symbol(i))
        if customize_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': customize_tokens})

    def __call__(self, item):
        self.pure_text_tokenizer = PureTextTokenizer()
        item = ''.join(next(self.pure_text_tokenizer([item])))
        return self.tokenizer(item, truncation=True, padding=True)

    def tokenize(self, item):
        self.pure_text_tokenizer = PureTextTokenizer()
        item = ''.join(next(self.pure_text_tokenizer([item])))
        return self.tokenizer.tokenize(item)


class FinetuneDataset(Dataset):
    def __init__(self, items):
        self.items = items
        self.len = len(items)

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return self.len


def finetune_bert(items, output_dir, pretrain_model="bert-base-chinese", train_params=None):
    model = AutoModelForMaskedLM.from_pretrained(pretrain_model)
    tokenizer = BertTokenizer(pretrain_model)
    # resize embedding for additional sepecial tokens
    model.resize_token_embeddings(len(tokenizer.tokenizer))

    # training parameters
    if train_params:
        mlm_probability = train_params['mlm_probability'] if 'mlm_probability' in train_params else 0.15
        epochs = train_params['epochs'] if 'epochs' in train_params else 1
        batch_size = train_params['batch_size'] if 'batch_size' in train_params else 64
        save_steps = train_params['save_steps'] if 'save_steps' in train_params else 100
        save_total_limit = train_params['save_total_limit'] if 'save_total_limit' in train_params else 2
        logging_steps = train_params['logging_steps'] if 'logging_steps' in train_params else 5
        gradient_accumulation_steps = train_params['gradient_accumulation_steps'] \
            if 'gradient_accumulation_steps' in train_params else 1
    else:
        # default
        mlm_probability = 0.15
        epochs = 1
        batch_size = 64
        save_steps = 1000
        save_total_limit = 2
        logging_steps = 5
        gradient_accumulation_steps = 1

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer.tokenizer, mlm=True, mlm_probability=mlm_probability
    )

    dataset = FinetuneDataset(items)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer.tokenizer,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)
