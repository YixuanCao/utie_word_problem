# coding=utf8

import re
from utie.common import Replacer


class Text2Int:

    units = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
    ]

    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    scales = ["hundred", "thousand", "million", "billion", "trillion"]
    numwords = {}
    for idx, word in enumerate(units):    numwords[word] = (1, idx)
    for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
    for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)
    keys = list(numwords.keys())
    for w in keys:
        if not w:
            continue
        w2 = w[0].upper() + w[1:]
        numwords[w2] = numwords[w]
    NUMBERS = list(numwords.keys())
    NUMBERS.remove('')
    numwords["and"] = (1, 0)
    numwords["-"] = (1, 0)


    @staticmethod
    def text2int(textnum):
        """
        把 Two hundred and fifty - four 变成 254
        :param textnum: 英文文本表示的整数
        :return:
        """
        current = result = 0
        words = textnum.split()  # 单独出现million等词时不处理，有少量数据中会在公式中把‘2 million’ 转化为2000000
        if len(words) == 1 and words[0] in Text2Int.scales:
            return words[0]
        for word in textnum.split():
            if word not in Text2Int.numwords:
                raise Exception("Illegal word: " + word)

            scale, increment = Text2Int.numwords[word]
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0

        return result + current


class SpecialNums:
    """
    处理 带分数  11,222 这样的带千分号的数
    """
    frac = r'(\d+)\s?/\s?(\d+)'
    frac_re = re.compile(frac)
    real_num = '\d+\.?\d*'
    percent_re = re.compile(f'({real_num})\s?(?:%|percent)')
    comma_in_num_re = re.compile(r'(\d+),(\d+)(\.\d+)?')
    mixed_number_re = re.compile(f'(\d+)\s+{frac}(?!%)')  # ‘33 1/3%’的情况不算在内

    @staticmethod
    def remove_comma_in_num(text):
        # 把 12,345 变成 12345
        def comma_f(m):
            groups = m.groups()
            if groups[-1]:
                return '{}{}{}'.format(groups[0], groups[1], groups[2])
            else:
                return '{}{}'.format(groups[0], groups[1])
        for i in range(text.count(',')):
            text = SpecialNums.comma_in_num_re.sub(repl=comma_f, string=text)
        return text

    @staticmethod
    def mixed_number_to_fraction(text):
        """ 3 1/3 -> 10/3"""
        def f(m):
            n, numerator, denominator = m.groups()
            return '{}/{}'.format(str(int(n) * int(denominator) + int(numerator)), denominator)
        return SpecialNums.mixed_number_re.sub(f, text)

    @staticmethod
    def percentage_to_fraction(text):
        def f(m):
            numerator = m.groups()[0]
            return '{}/100'.format(numerator)
        return SpecialNums.percent_re.sub(f, text)

    @staticmethod
    def handle(text, no_percent):
        text = SpecialNums.remove_comma_in_num(text)
        text = SpecialNums.mixed_number_to_fraction(text)
        if not no_percent:
            text = SpecialNums.percentage_to_fraction(text)
        return text

    @staticmethod
    def parse_frac(t):
        m = SpecialNums.frac_re.match(t)
        if m:
            return m.groups()
        return None

    @staticmethod
    def is_frac(t):
        if SpecialNums.frac_re.match(t):
            return True
        return False


class EnglishToNumber:
    """
    英文描述的数字变成数字
    如 two -> 2 , one twentieth -> 1/20
    """

    # puncts = re.compile(r'[\.\s,\?\n\r\-]')
    puncts = re.compile(r'\.\s|,\s|\?|\n|\r|(?<=[\w)])-|\s|\+|=|\$|\(|\)')

    multiple_replace_dict = {
        'twice': '2 times',
        'thrice': '3 times',
        'doubled': 'timed by 2',
        'double': '2 time',
        'doubles': '2 times',
        'twine': '2 times',
        'tripled': 'timed by 3',
        'triples': '3 times'
    }
    fraction_replace_dict = {
        'third': '3',
        'forth': '4',
        'fourth': '4',
        'quarter': '4',  # quarter 也有钱的意思
        'fifth': '5',
        'sixth': '6',
        'seventh': '7',
        'eighth': '8',
        'ninth': '9',
        'tenth': '10',
        'twentieth': '20',
        'thirtieth': '30',
        'fortieth': '40',
        'fourtieth': '40',
        'fiftieth': '50',
        'sixtieth': '60',
        'seventieth': '70',
        'eightieth': '80',
        'ninetieth': '90',
        'hundredth': '100'
    }

    multiple_re = re.compile('twine|twice|thrice|double[ds]?|triple[ds]?')
    fraction_re = re.compile('(?:1\s|1|)(half)|(\d+)\s?(third|fou?rth|quarter|fifth|sixth|seventh|eighth|ninths|'
                             'tenth|twentieth|thirtieth|fou?rtieth|fiftieth|sixtieth|hundredth)s?')

    @staticmethod
    def handle(text, no_percent):
        text = EnglishToNumber.english_phrase_to_num(text)
        text = EnglishToNumber.english_multiple_fraction_to_num(text)
        text = EnglishToNumber.pro_handle(text, no_percent)
        return text

    @staticmethod
    def pro_handle(text, no_percent):
        if not no_percent:
            text = text.replace('negative ', '-')
        else:
            text = text.replace('negative', 'negative -1').replace('2 numbers', 'two numbers').replace('1 number', 'one number')
        segs = text.strip().split()
        new_segs = []
        for seg in segs:
            nums = _extract_nums(seg, no_percent)
            if nums:
                pre_end = 0
                for i, n in nums:
                    if pre_end < n[0]:
                        new_segs.append(seg[pre_end:n[0]])
                    new_segs.append(seg[n[0]:n[1]])
                    pre_end = n[1]
                if pre_end < len(seg):
                    new_segs.append(seg[pre_end:])
            else:
                new_segs.append(seg)
        return ' '.join(new_segs)

    @staticmethod
    def split_by_punc(text):
        def f(t):
            """ 空格不变，其他的在两侧加空格"""
            t = t.group()
            if t != ' ':
                return f' {t} '
            else:
                return t
        words = EnglishToNumber.puncts.sub(f, text).split()
        return words

    @staticmethod
    def english_phrase_to_num(text):
        """
        所有单词按空格、标点符号隔开，每个数字作为一个单词，把英语的 twenty two 变成 22
        :param text:
        :return: 空格隔开的单词
        """
        words = EnglishToNumber.split_by_punc(text + ' ')  # 不加个' '的话最后一个词会跟末尾的'.'连在一块
        phrase = []  # (word, i)
        phrase_to_digit = {}  # { (i, j, k): 111 }
        new_words = []
        for i, word in enumerate(words + ['x']):
            if word in Text2Int.NUMBERS or (phrase and word in ['and', '-']):
                phrase.append([word, i])
            else:
                if phrase:
                    try:
                        pwords, pindexes = zip(*phrase)
                        dig = Text2Int.text2int(u' '.join(pwords))
                        phrase_to_digit[pindexes] = dig
                        new_words.append(str(dig))
                        phrase = []
                    except Exception as e:
                        print(e)
                        print(u'error in {}'.format(text))
                        return text
                if i < len(words):
                    new_words.append(word)
        return ' '.join(new_words)

    @staticmethod
    def english_multiple_fraction_to_num(text):
        """
        twice,  one half -> 2 time, 1/2
        :param text:
        :return:
        """
        def multiple_f(m):
            # print(m)
            matched = m.group()
            return EnglishToNumber.multiple_replace_dict[matched]

        def fraction_f(m):
            matched = m.groups()
            if matched[0]:
                return '1/2'
            if matched[2]:
                return '{}/{}'.format(matched[1], EnglishToNumber.fraction_replace_dict[matched[2]])
        text = EnglishToNumber.multiple_re.sub(multiple_f, text)
        text = EnglishToNumber.fraction_re.sub(fraction_f, text)
        return text


frac = r'\d+\.?\d*\s?/\s?\d+'  # 更新后能提取‘3.5/100’的情况
num_re = re.compile(f'\d+ {frac}|{frac}|\d+\.\d+\s?%|\d+\s?%|\d+\s?percent|\d+\.\d+|\d+|(?<![\w)])-\d+\.?\d*')
num_re_no_percent = re.compile(f'\d+ {frac}|{frac}|\d+\.\d+|\d+|(?<![\w)])-\d+\.?\d*')


def _extract_nums(text, no_percent):
    """在已经做过处理的文本上抽取数字"""
    if no_percent:
        num_matches = num_re_no_percent.finditer(text)
    else:
        num_matches = num_re.finditer(text)
    nums = []
    if num_matches:
        for m in num_matches:
            s, e = m.span()
            nums.append([text[s: e], (s, e)])
    return nums

