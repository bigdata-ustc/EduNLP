# Get Started

## 简介

EduNLP是一个专注于教育领域的语法语义分析工具包，可用于处理多模态的教育资源。EduNLP具有多模态结构识别，多模态分词、公式解析，语义向量化等功能，并提供多种预训练模型，是一种能用于实际教育产品的NLP工具。

## 基本概念

### item

传入的试题统称item，包括文本格式的填空题、字典格式的选择题等。

### 标准测试项目格式（SIF）

为了后续研究和使用的方便，规定的一个统一的试题语法标准。现已提供相关模块使得传入的item自动转换为符合SIF的格式。

### 语法解析

在教育资源中，文本、公式都具有内在的隐式或显式的语法结构，提取这种结构对后续进一步的处理是大有裨益的。

### 成分分解

由于教育资源是一种多模态数据，包含了诸如文本、图片、公式等数据结构； 同时在语义上也可能包含不同组成部分，例如题干、选项等，因此我们首先需要对教育资源的不同组成成分进行识别并进行分解。

### 令牌化

令牌化是自然语言处理中一项基本但是非常重要的步骤，它更令人为所熟知的名字是分句和分词。 在EduNLP中我们将令牌化分为不同的粒度。

### 向量化

将item、token转换为向量，得到最终产物。

## 处理流程

1.对传入的item进行解析，得到SIF格式；

2.对sif_item进行成分分解；

3.对经过成分分解的item进行令牌化；

4.使用已有或者使用提供的预训练模型，将令牌化后的item转换为向量。