import os
import numpy as np
from tqdm import tqdm
import jieba
import re
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from EduNLP.I2V import W2V, Bert, DisenQ
import warnings


VECTOR_MODEL_MAP = {
    "w2v": W2V,
    "bert": Bert,
    "disenq": DisenQ,
}

class PaperI2V():
    def __init__(self, pretrained_model_type, pretrained_model_dir, device="cpu", language=""):
        self.pretrained_model_type = pretrained_model_type
        self.pretrained_model_dir = pretrained_model_dir
        self.device = device
        
        tokenizer_kwargs = {"tokenizer_config_dir": pretrained_model_dir}
        
        if pretrained_model_type == "w2v":
            # set text tokenizer
            text_params = {
                "granularity": "word",
                "stopwords": None,
            }
            tokenizer_kwargs["text_params"] = text_params
            self.i2v = VECTOR_MODEL_MAP[pretrained_model_type]("pure_text",
                                                                'w2v',
                                                                pretrained_model_dir,
                                                                tokenizer_kwargs=tokenizer_kwargs
                                                                )
        elif pretrained_model_type in ["bert"]:
            self.i2v = VECTOR_MODEL_MAP[pretrained_model_type]('bert',
                                                                'bert',
                                                                pretrained_model_dir,
                                                                tokenizer_kwargs=tokenizer_kwargs,
                                                                device=device
                                                                )
        elif pretrained_model_type in ["disenq"]:
            self.i2v = VECTOR_MODEL_MAP[pretrained_model_type]('disenq',
                                                                'disenq',
                                                                pretrained_model_dir,
                                                                tokenizer_kwargs=tokenizer_kwargs,
                                                                device=device
                                                                )
    @classmethod
    def prcoess_line(cls, aline):
        return aline

    def to_embedding(self, aline):
        aline = self.prcoess_line(aline)
        
        if self.pretrained_model_type == "w2v_pub":
            words = jieba.lcut(aline)
            token_vector = []
            for word in words:
                if not re.sub(r'\s', '', word) == '':
                    temp_emb = self.i2v.word_to_embedding(word)
                    token_vector.append(temp_emb.tolist()) 
            token_vector = torch.FloatTensor(token_vector)
        elif self.pretrained_model_type == "w2v":
            token_vector = torch.FloatTensor(np.array(self.i2v.infer_token_vector([aline]))).squeeze(0)
        elif self.pretrained_model_type == "bert":
            token_vector = self.i2v.infer_token_vector([aline]).squeeze(0)
            token_vector = token_vector.float().cpu().detach()
        elif self.pretrained_model_type == "disenq":
            token_vector = self.i2v.infer_token_vector([aline]).squeeze(0)
            token_vector = token_vector.float().cpu().detach()

        if aline == "":
            warnings.warn("[ERROR] to_embedding:  aline is empty")
            return None

        return token_vector

class VecDataset(Dataset):
    def __init__(self,
                 language="",
                 text_data_dir=None,
                 emb_data_path=None,
                 pretrained_model_type="bert",
                 pretrained_model_dir=None,
                 paper_i2v: PaperI2V = None,
                 device="cpu",
                 mode="train",
                 do_w2v = False,
                ):
        self.device = device
        self.text_data_dir = text_data_dir
        self.emb_data_path = emb_data_path
        self.pretrained_model_type = pretrained_model_type
        self.pretrained_model_dir = pretrained_model_dir
        self.mode = mode
        self.input_data = []
        
        if paper_i2v is not None:
            self.paper_i2v = paper_i2v
        else:
            # 适配 双语pub_w2v
            language = "english" if language == "english" else ""
            self.paper_i2v = PaperI2V(pretrained_model_type,
                                    pretrained_model_dir,
                                    device=self.device,
                                    language=language)
        
        if not os.path.exists(emb_data_path) or do_w2v:
            os.makedirs(os.path.dirname(emb_data_path), exist_ok=True)
            self.set_all_text_embedding(text_data_dir, emb_data_path)
        else:
            self.get_all_text_embedding(emb_data_path)

    @property
    def embed_dim(self):
        return self.paper_i2v.vector_size

    def __getitem__(self, index):
        # return self.input_data[index]
        doc, tag = self.input_data[index]
        self.input_data[index][0] = [ sent.to(self.device) for sent in doc]
        self.input_data[index][1] = tag.to(self.device)
        return self.input_data[index]
    
    def __len__(self):
        return len(self.input_data)
    
    def set_all_text_embedding(self, indir, outpath):
        print(f'setting {self.mode} data ... ')
        path_list = os.listdir(indir)
        for file_name in tqdm(path_list):
            file_path = os.path.join(indir, file_name)
            doc, tag = self.paper_i2v.get_tagged_text_to_embedding(file_path)
            self.input_data.append( [doc, tag] )
        torch.save(self.input_data, outpath)

    def get_all_text_embedding(self, inpath):
        print(f'loading {self.mode} data ... ')
        self.input_data = torch.load(inpath)
    
    def pad(self, tags):
        max_length = max([_tags.size()[0] for _tags in tags])
        for i, _tags in enumerate(tags):
            _length = _tags.size()[0]
            tags[i] = F.pad(_tags, (0, max_length - _length))  # (max_length)
        return torch.stack(tags, dim=0)

    def collcate_fn(self, batch_data):
        documents, tags = list(zip(*batch_data))
        batch = {
            "documents": list(documents),
            "tags": self.pad(list(tags)),
        }
        return batch