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
    nums = f'\d+\s+{frac}|{frac}|{r}\s?%|{r}|\d+|(?<![\w)])-{r}'
    variables = '[a-z]'

    nums_re = re.compile(nums)
    variables_re = re.compile(variables)
    ops_re = re.compile(ops)
    ele_re = re.compile(f'{nums}|{ops}|{variables}')

    @staticmethod
    def add_mul_in_eq(text):
        # 把公式中省略的*加上
        def mul_f(m):
            groups = m.groups()
            return '{}*{}'.format(groups[0], groups[1])
        omit1 = re.compile('(\d+)([a-z])')
        omit2 = re.compile('(\d+|[a-z])(\()')
        omit3 = re.compile('(\))(\d+|[a-z])')
        omit4 = re.compile('(\))(\()')
        text = omit1.sub(repl=mul_f, string=text)
        text = omit2.sub(repl=mul_f, string=text)
        text = omit3.sub(repl=mul_f, string=text)
        text = omit4.sub(repl=mul_f, string=text)
        return text

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
        t = EqParse.add_mul_in_eq(t)
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
            print('parsed not equal to original: ')  # 下划线_ 如is_digit(x);百分号% 如 m % 2 = 1;x=71°18'-47°29';绝对值|;√;!
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
            unkns = l.split(':')[1].strip().split(',')
            unkns = [unk.strip() for unk in unkns]
        elif l.startswith('equ:'):
            eq = l.split(':')[1].strip()
            s, eq = EqParse.parse(eq, nums_in_text, v=True)
            if s:
                equations.append(eq)
                for e in eq:
                    # 公式中经常会出现未声明的变量
                    if e.isalpha() and e not in unkns:
                        unkns.append(e)
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
    text = text.replace('half m ', 'm/2 ').replace('7 and a half', '7.5 ').replace('39 cents', '0.39')  # 处理了几个特殊情况
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
        self.imp = []  # 用来统计公式中出现但文本中未出现数字的数据
        if not qa['equations']:
            self.success = False
        else:
            self.pair = self.parse()

    @staticmethod
    def text_implied_nums(text):
        ele_in_text = set()
        ele_in_text.add('1')
        ele_in_text.add('2')
        ele_in_text.add('0')
        ele_in_text.add('3.14')
        ele_in_text.add('3')
        ele_in_text.add('4')
        ele_in_text.add('5')
        if 'consecutive' in text:
            ele_in_text.add('2')
            ele_in_text.add('1')
        if contain_any(text, ['odd', 'even']):
            ele_in_text.add('2')
        if contain_any(text, ['percent', 'hundred', '%', 'cents', 'penny']):
            ele_in_text.add('100')
        if contain_any(text, ['square']):
            ele_in_text.add('2')
        if contain_any(text, ['hexa']):
            ele_in_text.add('6')
        if contain_any(text, [' min', 'second', ' mph', '/min', 'hour']):
            ele_in_text.add('60')
        if contain_any(text, ['digit']) or re.search('\d+\s?(mm|dm)', text):
            ele_in_text.add('10')
        if contain_any(text, ['opposite']):
            ele_in_text.add('-1')
        if contain_any(text, ['complementary']):
            ele_in_text.add('90')
        if contain_any(text, ['supplementary']):
            ele_in_text.add('180')
        if contain_any(text, ['month', 'year', 'annual']):
            ele_in_text.add('12')
            if contain_any(text, ['day']):
                ele_in_text.add('30')
        if contain_any(text, ['km', 'kg', 'milli', 'kilo', 'ml', 'mg', 'liter', 'litre']):
            ele_in_text.add('1000')
        # if re.search('\d+\s?(km|kg|milli|kilo|ml|mg)', text) or contain_any(text, ['milli', 'kilo']):
        if re.search('\d+\s?(cm|m)', text) or contain_any(text, ['centimeter']):
            ele_in_text.add('100')
        if 'feet' in text:
            if 'inch' in text:
                ele_in_text.add('12')
            elif 'yard' in text:
                ele_in_text.add('3')
        if re.search('[a-z]%', text):
            ele_in_text.add('0.01')
        return ele_in_text

    def align_nums(self, text_nums, eq_eles, implied_nums):
        eq_eles = list(set(eq_eles))
        text_num_freq = Counter(text_nums)
        implied_nums = [round(to_float(x), 5) for x in implied_nums]
        unified_text_nums = [round(to_float(x), 5) for x in text_nums]
        unieid_text_num_freq_with_imply = Counter(unified_text_nums + implied_nums)
        for ele in eq_eles:
            if EqParse.ele_type(ele) == 'num':
                freq = text_num_freq[ele]
                if freq > 1:
                    print('more than one match {} {}'.format(ele, text_nums))
                    print(self.qa)
                    return None
                if freq == 0:
                    if unieid_text_num_freq_with_imply[round(to_float(ele), 5)] == 0:
                        print(unieid_text_num_freq_with_imply)
                        print('not find {} {}'.format(ele, text_nums + implied_nums))
                        return None
        print('successfully matched')
        return True

    @staticmethod
    def transfer_num(new_text, eqs, numbers_in_eqs, unknowns):  # transfer num into "n"
        numbers_in_eqs = list(numbers_in_eqs)
        float_numbers_in_eqs = [round(to_float(n), 5) for n in numbers_in_eqs]
        segs = new_text.strip().split()
        new_unkowns = ['x', 'y', 'z']
        try:
            unk_dict = {unk: new_unkowns[i] for i, unk in enumerate(unknowns)}
        except:
            print('the number of unknowns out of bound')
        new_segs = []
        num_pos = []
        eq_segs = []
        nums = []  # 既出现在文本中又出现在公式中的数字
        for i, s in enumerate(segs):
            # 句子中出现类似'm/10'这种公式中的部分内容处理不了
            if s in numbers_in_eqs:
                new_segs.append('n')
                num_pos.append(i)
                nums.append(s)
            elif _extract_nums(s) and round(to_float(s), 5) in float_numbers_in_eqs:
                new_segs.append('n')
                num_pos.append(i)
                nums.append(s)
            else:
                new_segs.append(s)
        float_nums = {to_float(n): str(i) for i, n in enumerate(nums)}
        for eq in eqs:
            eq_seg = []
            for e in eq:
                if e in numbers_in_eqs:
                    if to_float(e) in float_nums:
                        eq_seg.append('N{}'.format(float_nums[to_float(e)]))
                    else:
                        eq_seg.append(e)
                elif e in unk_dict:
                    eq_seg.append(unk_dict[e])
                else:
                    eq_seg.append(e)
            eq_segs.append(eq_seg)
        pair = (new_segs, eq_segs, nums, num_pos)
        return pair

    def parse(self):
        qa = self.qa
        new_text, nums = extract_numbers(qa['text'])
        # if ''.join(new_text.split()) != ''.join(s['text'].lower().split()):
        #     print(new_text)
        #     print(s['text'])
        s, unknowns, eqs = parse_answer_equations(qa['equations'], {x[0] for x in nums})
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
        implied_nums = QuestionAnswerParser.text_implied_nums(qa['text'].lower())
        numbers_in_text.update(implied_nums)
        numbers_in_text = {round((to_float(x)), 5) for x in numbers_in_text}  # might be incorrect  1.8/100!=0.018
        numbers_in_eqs = {round((to_float(x)), 5) for x in numbers_in_eqs}

        if not numbers_in_eqs.issubset(numbers_in_text):
            print(qa)
            print(numbers_in_eqs.difference(numbers_in_text), numbers_in_eqs, numbers_in_text)
            if len(numbers_in_eqs.difference(numbers_in_text)) == 1:
                self.imp.extend(list(numbers_in_eqs.difference(numbers_in_text)))
            print('----------')
            return None
        else:
            numbers_in_text = [x[0] for x in nums]
            numbers_in_eqs = set()
            for eq in eqs:
                if not ('>' in eq or '<' in eq):
                    numbers_in_eqs.update([ele for ele in eq if EqParse.ele_type(ele) == 'num'])
            if not self.align_nums(numbers_in_text, numbers_in_eqs, list(implied_nums)):
                return None
            else:
                pair = QuestionAnswerParser.transfer_num(new_text, eqs, numbers_in_eqs, unknowns)
                return pair

operators = set('^+-*/><=()')


def op1_lower_2(op1, op2):
    # 返回 op1 优先级是否低于 op2
    pdict = {
        '(': -1,
        ')': -1,
        '^': 0,
        '*': 1,
        '/': 1,
        '+': 2,
        '-': 2
    }
    if pdict[op1] > pdict[op2]:
        return True
    return False


def infix_to_postfix(infix_expression):
    # 优先级列表
    # 初始化堆栈和后缀表达式
    stack = []
    postfix_expression = []
    for ele in infix_expression:
        # 操作数字符直接加入堆栈
        if ele not in operators:
            operand = ele
            postfix_expression.append(operand)
        else:
            if stack:  # 堆栈列表不为空
                if ele == '(':  # '('直接加入堆栈
                    stack.append(ele)
                elif ele == ')':
                    while stack[-1] != '(':
                        postfix_expression.append(stack.pop())
                    stack.pop()
                # 处理处于优先级1的运算符
                else:
                    while True:
                        if len(stack) == 0 or op1_lower_2(stack[-1], ele) or stack[-1] == '(':
                            stack.append(ele)
                            break
                        else:
                            postfix_expression.append(stack.pop())
            # 堆栈列表为空，直接加入堆栈
            else:
                stack.append(ele)
    # 获得最终的后缀表达式
    for i in range(len(stack)):
        postfix_expression.append(stack.pop())
    # print(postfix_expression)
    return postfix_expression


def eval_postfix(postfix_expression):
    # 利用后缀表达式计算结果
    stack = []
    for ele in postfix_expression:
        if ele not in operators:
            stack.append(ele)
        else:
            oper1 = float(stack.pop())
            oper2 = float(stack.pop())
            if ele == '+':
                stack.append(oper2 + oper1)
            elif ele == '-':
                stack.append(oper2 - oper1)
            elif ele == '*':
                stack.append(oper2 * oper1)
            elif ele == '/':
                stack.append(oper2 / oper1)
            elif ele == '^':
                stack.append(oper2 ** oper1)
            else:
                print('not a valid operator', ele)
    # print('后缀表达式:', " ".join(postfix_expression))
    # print('计算结果:', stack[0])
    return stack[0]


class Tree:
    def __init__(self, name, left, right, t):
        self.name = 'e{}'.format(name)  # id
        self.left = left  # value / another tree
        self.right = right
        self.t = t  # +-*/

    def is_leaf(self):
        return not isinstance(self.left, Tree) and not isinstance(self.right, Tree)

    def traverse(self):
        node_list = []
        if isinstance(self.left, Tree):
            left = self.left.name
            node_list.extend(self.left.traverse())
        else:
            left = self.left
        if isinstance(self.right, Tree):
            node_list.extend(self.right.traverse())
            right = self.right.name
        else:
            right = self.right
        node_list.append([self.name, left, right, self.t])
        return node_list


def postfix_to_tree(postfix_expression, rel):
    # 利用后缀表达式计算结果
    stack = []
    i = rel
    for ele in postfix_expression:
        if ele not in operators:
            stack.append(ele)
        else:
            oper1 = stack.pop()
            oper2 = stack.pop()
            node = Tree(i, oper2, oper1, ele)
            i += 1
            stack.append(node)
    # print('后缀表达式:', " ".join(postfix_expression))
    # print('计算结果:', stack[0])
    assert (len(stack) == 1)
    return stack[0]


def infix_to_relation_with_equal(index, rel_nums, infix_expression):
    # 中括号变成小括号
    replace_dict = {'[': '(', ']': ')'}
    infix_expression = [replace_dict.get(z, z) for z in infix_expression]
    assert infix_expression.count('=') == 1, ('Infix Expression is wrong :', infix_expression)
    infixs = []
    for i, ele in enumerate(infix_expression):
        if ele == '=':
            infixs.append(infix_expression[:i])
            infixs.append(infix_expression[i+1:])
    relations = generate_relations(index, rel_nums, infixs)
    return relations


def generate_relations(index, rel_nums, infixs):
    rel_nums
    relations = []
    tree = postfix_to_tree(infix_to_postfix(infixs[0]), rel_nums)
    if isinstance(tree, Tree):
        for n in tree.traverse():
            relations.append({'id': n[0], 'operands': [n[1], n[2]], 'type': n[-1]})
            rel_nums += 1
        left = relations[-1]['id']
    else:
        left = tree
    tree = postfix_to_tree(infix_to_postfix(infixs[1]), rel_nums)
    if isinstance(tree, Tree):
        for n in tree.traverse():
            relations.append({'id': n[0], 'operands': [n[1], n[2]], 'type': n[-1]})
            rel_nums += 1
        right = relations[-1]['id']
    else:
        right = tree
    relations.append({'id': 'eq' + str(index), 'operands': [left, right], 'type': '='})
    return relations

