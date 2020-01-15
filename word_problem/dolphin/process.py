# coding=utf8
import re
from collections import Counter

from word_problem.dolphin.text_num import _extract_nums, SpecialNums, EnglishToNumber
from word_problem.math23k.process import to_float


def remove_space(t):
    return ''.join(t.split())


class EqParse:
    r = '\d+\.?\d*'
    frac = '{r}\s?/\s?{r}'
    ops = '[\+\-\*\/\=\>\<\(\)\^]|log'
    nums = f'\d+\s+{frac}|{frac}|{r}\s?%|{r}|\d+'
    variables = '[abcdefghijklmnopqrstuvwxyz]'

    nums_re = re.compile(nums)
    variables_re = re.compile(variables)
    ops_re = re.compile(ops)
    ele_re = re.compile(f'{nums}|{ops}|{variables}')

    @staticmethod
    def parse(t, nums_in_text=None, v=False, only_num=False):
        """给定一个公式，进行解析，
        输入 2*y + 12 = 5*x
        返回 2, *, y, +, 12, =, 5, *, x
        输入 1/3 + 2/5 + 10%
        返回 1/3, +, 2/5, +, 10%
        :param v:
        :param t:
        :param nums_in_text: 文本中出现的数字
        :param only_num: True 只解析数字，不解析操作符、变量
        """
        success = True
        t = SpecialNums.handle(t)
        components = []
        if only_num:
            for m in EqParse.nums_re.finditer(t):
                s, e = m.span()
                components.append(t[s: e])
        else:
            for m in EqParse.ele_re.finditer(t):
                s, e = m.span()
                components.append(t[s: e])
            if nums_in_text:
                components = EqParse().merge_fractions(components, nums_in_text)
        if v and remove_space(''.join(components)) != remove_space(t):
            print('parsed not equal to original: ')
            print(t, components)
            success = False
            # print(components)
        return success, components

    @staticmethod
    def ele_type(t):
        if EqParse.nums_re.match(t):
            return 'num'
        if EqParse.ops_re.match(t):
            return 'op'
        if EqParse.variables_re.match(t):
            return 'var'

    @staticmethod
    def merge_fractions(components, nums_in_text):
        """
        解析出来的 components 如 x * 1 / 3 = 5，实际上文本中出现了 1/3 这个分数
        需要把components  改为  x * 1/3 = 5
        :param components:
        :param nums_in_text: 文本中出现的所有数字
        :return:
        """
        fractions_in_text = [n for n in nums_in_text if SpecialNums.is_frac(n)]
        for f in fractions_in_text:
            num, denom = SpecialNums.parse_frac(f)
            components = EqParse.merge_frac(components, num, denom)
        return components

    @staticmethod
    def merge_frac(components, numerator, denominator):
        merge_spans = []
        for i, ele in enumerate(components):
            if ele == '/':
                if components[i-1] == numerator and components[i+1] == denominator:
                    merge_spans.append([i-1, i+1])
        if not merge_spans:
            return components
        new_components = components[:merge_spans[0][0]]
        for i, (start, end) in enumerate(merge_spans):
            new_frac = '{}/{}'.format(components[start], components[end])
            new_components.append(new_frac)
            if i < len(merge_spans) - 1:
                next_start = merge_spans[i + 1][0]
                new_components.extend(components[end+1: next_start])
            else:
                new_components.extend(components[end+1: ])
        return new_components


def parse_answer_equations(answer_equations, nums_in_text):
    """
    解析答案里的公式字符串
    :param answer_equations:
    :param nums_in_text: 文本中出现的所有数字
    :return: success, unknowns, equations
    """
    unkns = []
    equations = []
    for l in answer_equations.split('\n'):
        l = l.strip()
        if l.startswith('unkn:'):
            unkns = l.split(':')[1].split(',')
        elif l.startswith('equ:'):
            eq = l.split(':')[1]
            s, eq = EqParse.parse(eq, nums_in_text, v=True)
            if s:
                equations.append(eq)
            else:
                return False, [], []
    return True, unkns, equations


def extract_numbers(text):
    """
    识别出所有的数字，记录其位置（空格分割的word index），对应的数字
    :param text: 题目
    :return text, [['300', (46, 49)], ['40', (71, 73)], ['8', (136, 137)]]

    """
    text = text.lower()
    text = SpecialNums.handle(text)
    text = EnglishToNumber.handle(text)
    nums = _extract_nums(text)
    return text, nums


def contain_any(text, eles):
    for ele in eles:
        if ele in text:
            return True
    return False


class QuestionAnswerParser:
    def __init__(self, qa):
        """

        :param qa:
        {'original_text': '',
         'text': 'One number is 3 less than a second number. Twice the second number is 12 less than 5 times the first. Find the two numbers.',
         'sources': ['https://answers.yahoo.com/question/index?qid=20091227181039AAZHeN8'],
         'flag': 0,
         'ans': '6; 9',
         'equations': 'unkn: x,y\r\nequ: x + 3 = y\r\nequ: 2*y + 12 = 5*x',
         'id': 'yahoo.answers.20091227181039aazhen8'}
        """
        self.qa = qa
        self.success = True

        if not qa['equations']:
            self.success = False
        else:
            self.parse()

    @staticmethod
    def text_implied_nums(new_text):
        ele_in_text = set()
        ele_in_text.add('1')
        if 'consecutive' in new_text:
            ele_in_text.add('2')
            ele_in_text.add('1')
        if contain_any(new_text, ['odd', 'even']):
            ele_in_text.add('2')
        if contain_any(new_text, ['percent', 'hundred', '%']):
            ele_in_text.add('100')
        if contain_any(new_text, ['square']):
            ele_in_text.add('2')
        if contain_any(new_text, ['digit']):
            ele_in_text.add('10')
        return ele_in_text

    def align_nums(self, text_nums, eq_eles):
        eq_eles = list(set(eq_eles))
        text_num_freq = Counter(text_nums)
        unified_text_nums = [str(to_float(x)) for x in text_nums]
        unieid_text_num_freq = Counter(unified_text_nums)
        for ele in eq_eles:
            if EqParse.ele_type(ele) == 'num':
                freq = text_num_freq[ele]
                if freq > 1:
                    print('more than one match {} {}'.format(ele, text_nums))
                    return None
                if freq == 0:
                    if unieid_text_num_freq[str(to_float(ele))] != 1:
                        print('not find {} {}'.format(ele, text_nums))
                        return None
        print('successfully matched')
        return True

    def parse(self):
        qa = self.qa
        new_text, nums = extract_numbers(qa['text'])
        # if ''.join(new_text.split()) != ''.join(s['text'].lower().split()):
        #     print(new_text)
        #     print(s['text'])
        s, unkowns, eqs = parse_answer_equations(qa['equations'], {x[0] for x in nums})
        numbers_in_eqs = set()
        for eq in eqs:
            if not ('>' in eq or '<' in eq):
                numbers_in_eqs.update([ele for ele in eq if EqParse.ele_type(ele) == 'num'])

        numbers_in_text = {x[0] for x in nums}
        # for ele in list(ele_in_text):
        #     if ele.endswith('%'):
        #         ele_in_text.add(ele[:-1])
        #         ele_in_text.add(str(to_float(ele)))
        # ele_in_text.update(QuestionAnswerParser.text_implied_nums(new_text))

        # '    the sum of a number plus 1/2 of itself is 18 . 75 . find the number .'
        implied_nums = QuestionAnswerParser.text_implied_nums(new_text)
        numbers_in_text.update(implied_nums)
        numbers_in_text = {str(to_float(x)) for x in numbers_in_text}  # might be incorrect  1.8/100!=0.018
        numbers_in_eqs = {str(to_float(x)) for x in numbers_in_eqs}

        if not numbers_in_eqs.issubset(numbers_in_text):
            print(qa['text'])
            print(new_text)
            print(eqs)
            print(numbers_in_eqs.difference(numbers_in_text), numbers_in_eqs, numbers_in_text)
            print('----------')
        else:
            numbers_in_text = [x[0] for x in nums] + list(implied_nums)
            numbers_in_eqs = set()
            for eq in eqs:
                if not ('>' in eq or '<' in eq):
                    numbers_in_eqs.update([ele for ele in eq if EqParse.ele_type(ele) == 'num'])
            if not self.align_nums(numbers_in_text, numbers_in_eqs):
                print(new_text)
