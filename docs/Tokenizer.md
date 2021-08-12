# Tokenizer

## 概述

为了方便后续向量化表征试题，本模块提供题目文本的令牌化解析（Tokenization），即将题目文本转换成令牌序列。按照试题文本类型分为以下四部分：

1. 分句（sentence-tokenization）：将较长的文档切分成若干句子的过程称为“分句”。每个句子为一个“令牌”（token）。（待实现）

1. 标记解析（text-tokenization）：一个句子（不含公式）是由若干“标记”按顺序构成的，将一个句子切分为若干标记的过程称为“标记解析”。每个标记为一个“令牌”（token）。
2. 公式解析（formula-tokenization）：理科类文本中常常含有公式。将一个符合 latex 语法的公式切分为标记字符列表的过程称为“公式解析”。每个标记字符为一个“令牌”（token）。
3. 综合解析（item-tokenization）：将带公式的句子切分为若干标记的过程。每个标记为一个“令牌”（token）。

目前提供以下模型：

- TextTokenizer

## TextTokenizer

此为一种综合解析的模型，使用线性分割的方法，先后通过segment（详细请见成分分解部分）和tokenize（详细请见令牌化部分）过程，最终得到词/字级别的切分结果。

### 具体流程主要为：

1.通过segment对结构成分进行分解，将items中元素按类型分为text、formula、figure、question mark、tag、sep。

2.对items中的各部分进行标签化操作，对text、formula部分不进行处理、其他部分转成特殊符号进行保留。

3.传入tokenize函数中，进行分词操作，得到最终结果。

### 处理的效果如下：

原始items： ["已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$"]

处理后的产物：['已知', '集合', 'A', '=', '\\left', '\\{', 'x', '\\mid', 'x', '^', '{', '2', '}', '-', '3', 'x', '-', '4', '<',

  '0', '\\right', '\\}', ',', '\\quad', 'B', '=', '\\{', '-', '4', ',', '1', ',', '3', ',', '5', '\\}', ',','\\quad', 'A', '\\cap', 'B', '=']