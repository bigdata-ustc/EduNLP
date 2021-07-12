from EduNLP.Formula.ast import str2ast, katex_parse


class Parser:
    def __init__(self, data):
        self.lookahead = 0
        self.head = 0
        self.text = data
        self.error_message = ''
        self.error_postion = 0
        self.error_flag = 0
        self.modify_flag = 0
        self.warnning = 0
        self.fomula_illegal_flag = 0
        self.fomula_illegal_message = ''

        # 定义特殊变量
        self.len_bracket = len('$\\SIFChoice$')
        self.len_underline = len('$\\SIFBlank$')

        # 定义 token
        self.error = -1
        self.character = 1
        self.en_pun = 2
        self.ch_pun = 3
        self.latex = 4
        self.end = 5
        self.empty = 6
        self.modify = 7
        self.blank = 8

        self.en_pun_list = [',', '.', '?', '!',
                            ':', ';', '\'', '\"', '(', ')', ' ', '_', '/', '|', '\\', '<', '>', '[', ']',
                            '-']  # add some other chars
        self.ch_pun_list = ['，', '。', '！', '？', '：',
                            '；', '‘', '’', '“', '”', '（', '）', ' ', '、', '《', '》', '—', '．']
        self.in_list = [',', '_', '-', '%']
        self.flag_list = ['，', '。', '！', '？', '：',
                          '；', '‘', '’', '“', '”', '（', '）', ' ', '、', '《', '》',
                          '$', ',', '.', '?', '!', ':', ';', '\'', '\"', '(', ')', ' ', '_', '/', '|', '<', '>', '-',
                          '[', ']', '—']

    def is_number(self, uchar):
        """判断一个unicode是否是数字"""
        if u'\u0030' <= uchar <= u'\u0039':
            # print(uchar, ord(uchar))(u'\u0030' <= uchar <= u'\u0039')
            return True
        else:
            return False

    def is_alphabet(self, uchar):
        """判断一个unicode是否是英文字母"""
        if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
            return True
        else:
            return False

    def is_chinese(self, uchar):
        """判断一个unicode是否是汉字"""
        if u'\u4e00' <= uchar <= u'\u9fa5':
            return True
        else:
            return False

    def _is_formula_legal(self, formula_str):
        r"""
        Judge whether the current formula meet our specification or not.

        Parameters
        ----------
        formula_str

        Returns
        -------
        True or False

        """
        legal_tags = ['FormFigureID', 'FormFigureBase64', 'FigureID', 'FigureBase64',
                      'SIFBlank', 'SIFChoice', 'SIFTag', 'SIFSep', 'SIFUnderline']
        for tag in legal_tags:
            if tag in formula_str:
                return True
        try:
            katex_parse(formula_str)
        except Exception as e:
            assert 'ParseError' in str(e)
            self.fomula_illegal_message = "[FormulaError] " + str(e)
            self.fomula_illegal_flag = 1
            return False
        return True

    def call_error(self):
        """语法解析函数"""
        # print('ERROR::position is >>> ',self.head)
        # print('ERROR::match is >>>', self.text[self.head])
        self.error_postion = self.head
        self.error_message = self.text[:self.head + 1]
        self.error_flag = 1

    def get_token(self):
        if self.head >= len(self.text):
            return self.empty
        ch = self.text[self.head]
        if self.is_chinese(ch):
            # 匹配中文字符 [\u4e00-\u9fa5]
            self.head += 1
            return self.character
        elif self.is_alphabet(ch):
            # 匹配公式之外的英文字母，只对两个汉字之间的字母做修正，其余匹配到的情况视为不合 latex 语法录入的公式
            left = head = self.head
            if self.head == 0:
                while (head < len(self.text) and (
                        self.is_alphabet(self.text[head]) or self.text[head] in self.in_list)):
                    head += 1
                if head == len(self.text) or self.is_chinese(self.text[head]) or self.text[head] in self.flag_list:
                    self.head = head
                    self.text = self.text[:left] + "$" + self.text[left:head] + "$" + self.text[head:]
                    self.head += 2
                    #                     print(self.text[left:self.head])
                    self.modify = 1
                    return self.modify
            else:
                forward = self.text[self.head - 1]
                if self.is_chinese(forward) or forward in self.flag_list:
                    while (head < len(self.text) and (
                            self.is_alphabet(self.text[head]) or self.text[head] in self.in_list)):
                        head += 1
                    if head == len(self.text) or self.is_chinese(self.text[head]) or self.text[head] in self.flag_list:
                        self.head = head
                        self.text = self.text[:left] + "$" + self.text[left:head] + "$" + self.text[head:]
                        self.head += 2
                        self.modify_flag = 1
                        return self.modify
            # self.call_error()
            # return self.error

        elif self.is_number(ch):
            # 匹配公式之外的数字，只对两个汉字之间的数字做修正，其余匹配到的情况视为不合 latex 语法录入的公式
            left = head = self.head
            if self.head == 0:
                while (head < len(self.text) and (
                        self.is_number(self.text[head]) or self.text[head] in self.in_list)):
                    head += 1
                if head == len(self.text) or self.is_chinese(self.text[head]) or self.text[head] in self.flag_list:
                    self.head = head
                    self.text = self.text[:left] + "$" + self.text[left:head] + "$" + self.text[head:]
                    self.head += 2
                    self.modify_flag = 1
                    return self.modify

            else:
                forward = self.text[self.head - 1]
                if self.is_chinese(forward) or forward in self.flag_list:
                    while (head < len(self.text) and (
                            self.is_number(self.text[head]) or self.text[head] in self.in_list)):
                        head += 1

                    if head == len(self.text) or self.is_chinese(self.text[head]) or self.text[head] in self.flag_list:
                        self.head = head
                        self.text = self.text[:left] + "$" + self.text[left:head] + "$" + self.text[head:]
                        self.head += 2
                        self.modify_flag = 1
                        return self.modify
            # self.call_error()
            # return self.error

        elif ch == '\n':
            # 匹配换行符
            self.head += 1
            return self.end

        elif ch in self.ch_pun_list:
            # 匹配中文标点
            left = self.head
            self.head += 1
            if self.text[left] == '（':
                # 匹配到一个左括号
                while self.text[self.head] == ' ' or self.text[self.head] == '\xa0':
                    self.head += 1
                if self.text[self.head] == '）':
                    self.head += 1
                    self.text = self.text[:left] + '$\\SIFChoice$' + self.text[self.head:]
                    self.head += self.len_bracket
                    self.modify_flag = 1
                    return self.modify
            return self.ch_pun
        elif ch in self.en_pun_list:
            # 匹配英文标点
            # print('en-pun-list')
            left = self.head
            self.head += 1
            if self.text[left] == '(':
                # 匹配到一个左括号
                while self.text[self.head] == ' ' or self.text[self.head] == '\xa0':
                    self.head += 1
                if self.text[self.head] == ')':
                    self.head += 1
                    self.text = self.text[:left] + '$\\SIFChoice$' + self.text[self.head:]
                    self.head += self.len_bracket
                    self.modify_flag = 1
                    return self.modify
            if self.text[left] == '_':
                # 匹配到一个下划线
                # print('this is an underline')
                while self.text[self.head] == '_' or self.text[self.head] == ' ':
                    self.head += 1
                    if self.head >= len(self.text):
                        break
                # print('change the text')
                self.text = self.text[:left] + '$\\SIFBlank$' + self.text[self.head:]
                self.head += self.len_underline
                # print(self.text)
                self.modify_flag = 1
                return self.modify
            return self.en_pun

        elif ch == '$':
            # 匹配 latex 公式
            self.head += 1
            flag = 1
            formula_start = self.head
            while self.head < len(self.text) and self.text[self.head] != '$':
                ch_informula = self.text[self.head]
                if flag and self.is_chinese(ch_informula):
                    # latex 中出现中文字符，打印且只打印一次 warning
                    print("Warning: there is some chinese characters in formula!")
                    self.warnning = 1
                    flag = 0
                self.head += 1
            if self.head >= len(self.text):
                self.call_error()
                return self.error
            # 检查latex公式的完整性和可解析性
            if not self._is_formula_legal(self.text[formula_start:self.head]):
                self.call_error()
                return self.error
            self.head += 1
            # print('is latex!')
            return self.latex
        else:
            self.call_error()
            return self.error

    def next_token(self):
        #         print('call next_token')
        #         if self.error_flag:
        #             return
        self.lookahead = self.get_token()
        if self.error_flag:
            return

    def match(self, terminal):
        #         print('call match')
        # if self.error_flag:
        #     return
        if self.lookahead == terminal:
            self.next_token()
            if self.error_flag:
                return
        # else:
        #     print('match error!')
        #     self.call_error()

    def txt(self):
        #         print('call txt')
        #         if self.error_flag:
        #             return
        self.lookahead = self.get_token()
        if self.error_flag:
            return
        if self.lookahead == self.character or self.lookahead == self.en_pun or \
                self.lookahead == self.ch_pun or self.lookahead == self.latex:
            self.match(self.lookahead)

    def txt_list(self):
        #         print('call txt_list')
        #         if self.error_flag:
        #             return
        self.txt()
        if self.error_flag:
            return
        if self.lookahead != self.empty:
            self.txt_list()

    def description(self):
        #         print('call description')
        #         if self.error_flag:
        #             return
        self.txt_list()
        if self.error_flag:
            return
        if self.lookahead == self.empty:
            self.match(self.lookahead)

    def description_list(self):
        r"""
        use Parser to process and describe the txt

        Parameters
        ----------

        Returns
        ----------

        Examples
        --------
        >>> text = '生产某种零件的A工厂25名工人的日加工零件数_   _'
        >>> text_parser = Parser(text)
        >>> text_parser.description_list()
        >>> text_parser.text
        '生产某种零件的$A$工厂$25$名工人的日加工零件数$\\SIFBlank$'
        >>> text = 'X的分布列为(   )'
        >>> text_parser = Parser(text)
        >>> text_parser.description_list()
        >>> text_parser.text
        '$X$的分布列为$\\SIFChoice$'
        >>> text = '① AB是⊙O的直径，AC是⊙O的切线，BC交⊙O于点E．AC的中点为D'
        >>> text_parser = Parser(text)
        >>> text_parser.description_list()
        >>> text_parser.error_flag
        1
        >>> text = '支持公式如$\\frac{y}{x}$，$\\SIFBlank$，$\\FigureID{1}$，不支持公式如$\\frac{ \\dddot y}{x}$'
        >>> text_parser = Parser(text)
        >>> text_parser.description_list()
        >>> text_parser.fomula_illegal_flag
        1
        """
        # print('call description_list')
        self.description()
        if self.error_flag:
            # print("Error")
            return
        if self.lookahead != self.empty:
            self.description_list()  # pragma: no cover
        else:
            self.error_flag = 0
            # print('parse successfully!')
