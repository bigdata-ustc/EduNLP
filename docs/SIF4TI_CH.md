# 标准项目格式

version: 0.2

为了后续研究和使用的方便，我们需要一个统一的试题语法标准。

## 语法规则
1. 题目文本中只允许出现中文字符、中英文标点和换行符。
2. 使用 \$\SIFBlank\$ 替换横线，对于选择题中的括号使用 \$\SIFChoice\$ 替换。
3. 图片 ID 以公式的形式嵌入文本中：`$\FigureID{ uuid }$` 或用 base64 编码表示，特别的，内容为公式的图片用`$\FormFigureID{ uuid }$`表示。
4. 文本标注格式：统一用 `$\textf{item,CHAR_EN}$` 表示，目前定义的有：b-加粗，i-斜体，u-下划线，w-下划波浪线，d-加点，t-标题。标注可以混用，按字母顺序排序，例如：$\textf{EduNLP, biu}$ 表示 <u>***EduNLP***</u>
5. 其余诸如，英文字母、罗马字符、数字等数学符号一律需要使用 latex 格式表示，即嵌在 `$$` 之中。
6. 分子式的录入标准暂且参考 [INCHI](https://zh.wikipedia.org/wiki/%E5%9B%BD%E9%99%85%E5%8C%96%E5%90%88%E7%89%A9%E6%A0%87%E8%AF%86)
7. 目前对 latex 内部语法没有要求。

```
1. Item -> CHARACTER|EN_PUN_LIST|CH_PUN_LIST|FORMULA|QUES_MARK
2. EN_PUN_LIST -> [',', '.', '?', '!', ':', ';', '\'', '\"', '(', ')', ' ','_','/','|','\\','<','>','[',']','-']
3. CH_PUN_LIST -> ['，', '。', '！', '？', '：','；', '‘', '’', '“', '”', '（', '）', ' ', '、','《','》','—','．']
4. FORMULA -> $latex formula$ | $\FormFigureID{UUID}$ | $\FormFigureBase64{BASE64}$
5. FIGURE -> $\FigureID{UUID}$ | $\FigureBase64{BASE64}$
6. UUID -> [a-zA-Z\-0-9]+
7. CHARACTER -> CHAR_EN | CHAR_CH
8. CHAR_EN -> [a-zA-Z]+
9. CHAR_CH -> [\u4e00-\u9fa5]+
10. DIGITAL -> [0-9]+
11. QUES_MARK -> $\SIFBlank$ | $\SIFChoice$
```

### 注意事项
1. 保留字符与转义
2. 数字
3. 选空与填空
4. 对于单个的数字或字符也需要添加 `$$`（目前能实现自动校验）
5. latex 公式中尽量不出现中文：（`\text{这里出现中文}`）
6. MySql 数据库导入数据时会自动忽略一个 `\`，所以录入的公式需要进一步处理为 `\\`

## 示例

标准形式:

1. `若$x,y$满足约束条件$\\left\\{\\begin{array}{c}2 x+y-2 \\leq 0 \\\\ x-y-1 \\geq 0 \\\\ y+1 \\geq 0\\end{array}\\right.$，则$z=x+7 y$的最大值$\\SIFUnderline$'`

2. `已知函数$f(x)=|3 x+1|-2|x|$画出$y=f(x)$的图像求不等式$f(x)>f(x+1)$的解集$\\PictureID{3bf2ddf4-8af1-11eb-b750-b46bfc50aa29}$$\\PictureID{59b8bd14-8af1-11eb-93a5-b46bfc50aa29}$$\\PictureID{63118b3a-8b75-11eb-a5c0-b46bfc50aa29}$$\\PictureID{6a006179-8b76-11eb-b386-b46bfc50aa29}$$\\PictureID{088f15eb-8b7c-11eb-a86f-b46bfc50aa29}$`

非标准形式：

1. 字母、数字和数学符号连续混合出现：
    例如：
    `完成下面的2x2列联表，`
    `（单位：m3）`
    `则输出的n=`
    
2. 特殊的数学符号没有用 latex 公式表示：
    例如：
    `命题中真命题的序号是 ①`
    `AB是⊙O的直径，AC是⊙O的切线，BC交⊙O于点E．若D为AC的中点`
    
3. 出现以 unicode 编码写成的字符
    例如：`则$a$的取值范围是（\u3000\u3000）`


## Change Log

2021-05-18

修改：
1. 原用 \$\SIFUnderline\$ 和 \$\SIFBracket\$ 来替换填空题中的横线和选择题中的括号，现分别用 \$\SIFBlank\$ 和 \$\SIFChoice\$ 替换。 
2. 原统一用`$\PictureID{ uuid }$`表示图片，现使用`$\FigureID{ uuid }$`，其中对于数据公式，用`$\FormFigureID{ uuid }$`来表示。

2021-06-28 
  
添加： 
1. 注明 `$$` 之中不能出现换行符。 
2. 添加文本标注格式说明。 

