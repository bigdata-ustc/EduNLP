# 标准测试项目格式

为……（意义）

## 语法规则
1. 题目文本中只允许出现中文字符、中英文标点和换行符。
2. 使用 \$\LUNAUnderline\$ 替换横线，对于选择题中的括号使用 \$\LUNABrackets\$ 替换.
3. 图片 ID 以公式的形式嵌入文本中：`$\PictureID{ uuid1 }$`
4. 其余诸如，英文字母、罗马字符、数字等数学符号一律需要使用 latex 格式表示，即嵌在 `$$` 之中。
5. 目前对 latex 内部语法没有要求。

```
1. Item -> CHARACTER|EN_PUN_LIST|CH_PUN_LIST|FORMULA|SPECIAL_TOKEN
2. EN_PUN_LIST -> [',', '.', '?', '!', ':', ';', '\'', '\"', '(', ')', ' ','_','/','|','\\','<','>']
3. CH_PUN_LIST -> ['，', '。', '！', '？', '：','；', '‘', '’', '“', '”', '（', '）', ' ', '、','《','》']
4. FORMULA -> $latex formula$ | PICTURE
5. PICTURE -> $PICTUREID{DIGITAL}$
6. CHARACTER -> CHAR_EN | CHAR_CH
7. CHAR_EN -> [a-zA-Z]+
8. CHAR_CH -> []+
9. DIGITAL -> [0-9]+
10. SPECIAL_TOKEN -> $\SIFUnderline$ | $\SIFBrackets$
```

### 注意事项
1. 保留字符与转义
2. 数字
3. 选空与填空
