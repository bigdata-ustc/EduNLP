from EduNLP.Vector import get_pretrained_t2v
import json
from tqdm import tqdm
from EduNLP.SIF.segment import seg
from EduNLP.SIF.tokenization import tokenize
from EduNLP.Pretrain import GensimWordTokenizer

def load_items():
    test_items = [
        {'ques_content':'有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$,若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'},
        {'ques_content':'如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$,若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'},
        {'ques_content':'<div>Below is a discussion on a website.<br><table border=\1'},
    ]
    for line in test_items:
        yield line
    # """or use your jsonfile like this"""
    # work_file_path = "../../../data/OpenLUNA.json"
    # with open(work_file_path, 'r', encoding="utf-8") as f:
    #     for line in f:
    #         yield json.loads(line)


token_items = []
for item in tqdm(load_items(), "sifing"):
    # transform content into special marks('g','m','a','s'), except text('t') and formula('f').
    # 'general' means symbolize the Formula in figure format and use 'linear' method for formula segmentation
    tokenizer = GensimWordTokenizer(symbol="gmas", general=True)
    token_item = tokenizer(item["ques_content"])
    if token_item:
        token_items.append(token_item.tokens)

# make a model -> t2v
t2v = get_pretrained_t2v("d2v_eng_256")
print(t2v(token_items))