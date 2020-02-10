# coding=utf8
import collections
import re
from collections import Counter

from word_problem.dolphin.text_num import _extract_nums, SpecialNums, EnglishToNumber
from word_problem.math23k.process import to_float, Tree, infix_to_postfix, add_significant_number
from utie.data_label_to_candidate import reid

def remove_space(t):
    return ''.join(t.split())


class EqParse:
    r = '\d+\.?\d*'
    frac = '{r}\s?/\s?{r}'
    ops = '[\+\-\*\/\=\>\<\(\)\^]|log'
    nums = f'\d+\s+{frac}|{frac}|{r}\s?%|{r}|\d+|(?<![\w)])-{r}'
    variables = '[a-z_]+'

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
    def add_omit_zero(text):
        def omit_f(m):
            groups = m.groups()
            for i in groups:
                if i:
                    return '0{}'.format(i)
        omit_re = re.compile('(^\.\d)|((?<=\D)\.\d)')
        text = omit_re.sub(repl=omit_f, string=text)
        return text

    @staticmethod
    def parse(t, nums_in_text=None, v=False, only_num=False, no_percent=False):
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
        t = SpecialNums.handle(t, no_percent)
        t = EqParse.add_mul_in_eq(t)
        t = EqParse.add_omit_zero(t)
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


def parse_answer_equations(answer_equations, nums_in_text, no_percent):
    """
    解析答案里的公式字符串
    :param answer_equations:
    :param nums_in_text: 文本中出现的所有数字
    :return: success, unknowns, equations
    """
    unkns = []
    equations = []
    if isinstance(answer_equations, (list, tuple)):
        for eq in answer_equations:
            eq = eq.strip().lower()
            # if '=0' in eq:
            #     eq = eq.replace('=0', '').replace('-', '=')
            # elif '= 0' in eq:
            #     eq = eq.replace('= 0', '').replace('-', '=')
            s, eq = EqParse.parse(eq, nums_in_text, no_percent=no_percent, v=True)
            if s:
                equations.append(eq)
                unkns = {e for e in eq if e[0].isalpha()}
            else:
                return False, [], []

    else:
        for l in answer_equations.split('\n'):
            l = l.strip()
            if l.startswith('unkn:'):
                unkns = l.split(':')[1].strip().split(',')
                unkns = [unk.strip() for unk in unkns]
            elif l.startswith('equ:'):
                eq = l.split(':')[1].strip()
                s, eq = EqParse.parse(eq, nums_in_text, no_percent=no_percent, v=True,)
                if s:
                    equations.append(eq)
                    for e in eq:
                        # 公式中经常会出现未声明的变量
                        if e.isalpha() and e not in unkns:
                            unkns.append(e)
                else:
                    return False, [], []
    return True, unkns, equations


def extract_numbers(text, no_percent):
    """
    识别出所有的数字，记录其位置（空格分割的word index），对应的数字
    :param text: 题目
    :param no_percent: 是否把百分数提取出来， draw数据中为False
    :return text, [['300', (46, 49)], ['40', (71, 73)], ['8', (136, 137)]]

    """
    text = text.lower()
    text = text.replace('half m ', 'm/2 ').replace('7 and a half', '7.5 ').replace('39 cents', '0.39')  # 处理了几个特殊情况
    text = EqParse.add_omit_zero(text)
    text = SpecialNums.handle(text, no_percent)
    text = EnglishToNumber.handle(text, no_percent)
    nums = _extract_nums(text, no_percent)
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

    def gen_pair(self, no_percent=False):
        return self.parse(no_percent)

    @staticmethod
    def text_implied_nums(text):
        ele_in_text = set()
        ele_in_text.add('1')
        ele_in_text.add('2')
        # ele_in_text.add('0')
        if contain_any(text, ['radius', 'diameter', 'radians']):
            ele_in_text.add('3.14')
        # ele_in_text.add('3')
        # ele_in_text.add('4')
        if contain_any(text, ['quadrilateral']):
            ele_in_text.add('360')
        if 'three consecutive odd' in text:
            ele_in_text.add('5')
        if 'four consecutive' in text or 'consecutive odd' in text:
            ele_in_text.add('3')
        # if contain_any(text, ['odd', 'even']):
        #     ele_in_text.add('2')
        if contain_any(text, ['percent', 'hundred', '%', 'cents', 'penny']):
            ele_in_text.add('100')
        if contain_any(text, ['quarter', 'square', 'legs', 'rabbits']):
            ele_in_text.add('4')
        if contain_any(text, ['trike', ' 1/3 ', 'tripling', 'one third', ' cone ', 'pyramid']):
            ele_in_text.add('3')
        if contain_any(text, ['hexa']):
            ele_in_text.add('6')
        if contain_any(text, [' min', 'second', ' mph', '/min', 'hour']):
            ele_in_text.add('60')
        if contain_any(text, ['digit', 'tens']) or re.search('\d+\s?(mm|dm)', text):
            ele_in_text.add('10')
        # if contain_any(text, ['opposite']):
        #     ele_in_text.add('-1')
        if contain_any(text, ['complement']):
            ele_in_text.add('90')
        if contain_any(text, ['supplement', 'triangle', 'linear pair', 'trigangle', 'radians']):
            ele_in_text.add('180')
        if contain_any(text, ['month', 'year', 'annual']):
            ele_in_text.add('12')
            if contain_any(text, ['day']):
                ele_in_text.add('30')
        if contain_any(text, [' km ', 'kg', 'milli', 'kilo', 'ml', 'mg', 'liter', 'litre', 'thousands']):
            ele_in_text.add('1000')
        # if re.search('\d+\s?(km|kg|milli|kilo|ml|mg)', text) or contain_any(text, ['milli', 'kilo']):
        if re.search('\d+\s?(cm|m)', text) or contain_any(text, ['centimeter']):
            ele_in_text.add('100')
        if 'inch' in text:
                ele_in_text.add('12')
        if 'yard' in text and 'feet' in text:
            ele_in_text.add('3')
        # if re.search('[a-z]%', text):
        #     ele_in_text.add('0.01')
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
    def transfer_num(new_text, eqs, numbers_in_eqs, numbers_in_text, unknowns, no_percent):  # transfer num into "n"
        float_numbers_in_text = [round(to_float(n), 5) for n in numbers_in_text]
        segs = new_text.strip().split()
        new_unkowns = ['x', 'y', 'z']
        try:
            unk_dict = {unk: new_unkowns[i] for i, unk in enumerate(unknowns)}
        except:
            print('the number of unknowns out of bound', unknowns)
        new_segs = []
        eq_segs = []
        # nums_in_text = ['1', '2', '1', '4']
        nums_pos_dict = collections.OrderedDict()
        for num in float_numbers_in_text:
            if num not in nums_pos_dict:
                nums_pos_dict[num] = []

        for i, s in enumerate(segs):
            # 句子中出现类似'm/10'这种公式中的部分内容处理不了
            if s in numbers_in_text:
                new_segs.append('n')
                nums_pos_dict[round(to_float(s), 5)].append(str(i))
            elif _extract_nums(s, no_percent) and round(to_float(s), 5) in float_numbers_in_text:
                new_segs.append('n')
                nums_pos_dict[round(to_float(s), 5)].append(str(i))
            else:
                new_segs.append(s)
        nums_index_dict = {n: str(i) for i, n in enumerate(nums_pos_dict)}
        for eq in eqs:
            eq_seg = []
            for e in eq:
                if e in numbers_in_eqs:
                    if round(to_float(e), 5) in nums_index_dict:
                        eq_seg.append('N{}'.format(nums_index_dict[round(to_float(e), 5)]))
                    else:
                        eq_seg.append(e)
                elif e in unk_dict:
                    eq_seg.append(unk_dict[e])
                else:
                    eq_seg.append(e)
            eq_segs.append(eq_seg)
        return new_segs, eq_segs, numbers_in_text, nums_pos_dict

    def parse(self, no_percent=False):
        qa = self.qa
        new_text, nums = extract_numbers(qa['text'], no_percent)
        # if ''.join(new_text.split()) != ''.join(s['text'].lower().split()):
        #     print(new_text)
        #     print(s['text'])
        s, unknowns, eqs = parse_answer_equations(qa['equations'], {x[0] for x in nums}, no_percent)
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
                pair = QuestionAnswerParser.transfer_num(new_text, eqs, numbers_in_eqs, numbers_in_text, unknowns, no_percent)
                new_segs, eq_segs, nums, num_pos = pair
                return new_segs, eq_segs, nums, num_pos, list(implied_nums), qa['ans']


operators = set('^+-*/><=()')


def postfix_to_tree_with_equal(postfix_expression, rel):
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
    relations = []
    tree = postfix_to_tree_with_equal(infix_to_postfix(infixs[0]), rel_nums)
    if isinstance(tree, Tree):
        for n in tree.traverse():
            relations.append({'id': n[0], 'operands': [n[1], n[2]], 'type': n[-1]})
            rel_nums += 1
        left = relations[-1]['id']
    else:
        left = tree
    tree = postfix_to_tree_with_equal(infix_to_postfix(infixs[1]), rel_nums)
    if isinstance(tree, Tree):
        for n in tree.traverse():
            relations.append({'id': n[0], 'operands': [n[1], n[2]], 'type': n[-1]})
            rel_nums += 1
        right = relations[-1]['id']
    else:
        right = tree
    relations.append({'id': 'eq' + str(index), 'operands': [left, right], 'type': '='})
    return relations


def data_to_utie(data_path):
    import json
    data = json.load(open(data_path))
    pairs = get_pairs(data)
    utie_data = pairs_to_utie(pairs, 3)
    return utie_data


def get_pairs(data):
    pairs = {}
    for i, s in enumerate(data):
        if not (s['equations'] and s['text']):
            continue
        try:
            s = bad_data_handle(s)
            qa_p = QuestionAnswerParser(s)
            pair = qa_p.gen_pair()
            if pair[1]:
                pairs[s['id']] = pair
        except:
            pass
    print('{} samples, {} success to convert to pairs'.format(len(data), len(pairs)))
    return pairs


implied_words = [{'word': '1', 'id': 'sp1'},
                 {'word': '4', 'id': 'sp4'},
                 {'word': '5', 'id': 'sp5'},
                 {'word': '0', 'id': 'sp0'},
                 {'word': '6', 'id': 'sp6'},
                 {'word': '60', 'id': 'sp60'},
                 {'word': '-1', 'id': 'sp-1'},
                 {'word': '2', 'id': 'sp2'},
                 {'word': '100', 'id': 'sp100'},
                 {'word': '360', 'id': 'sp360'},
                 {'word': '1000', 'id': 'sp1000'},
                 {'word': '10', 'id': 'sp10'},
                 {'word': '3', 'id': 'sp3'},
                 {'word': '30', 'id': 'sp30'},
                 {'word': '12', 'id': 'sp12'},
                 {'word': '0.01', 'id': 'sp0.01'},
                 {'word': '90', 'id': 'sp90'},
                 {'word': '180', 'id': 'sp180'},
                 {'word': 'pi', 'id': 'sp_pi'}]
implied_ents = [{'id': '1', 'tokens': 'sp1', 'type': 'num'},
                {'id': '4', 'tokens': 'sp4', 'type': 'num'},
                {'id': '5', 'tokens': 'sp5', 'type': 'num'},
                {'id': '6', 'tokens': 'sp6', 'type': 'num'},
                {'id': '60', 'tokens': 'sp60', 'type': 'num'},
                {'id': '0', 'tokens': 'sp0', 'type': 'num'},
                {'id': '-1', 'tokens': 'sp-1', 'type': 'num'},
                {'id': '2', 'tokens': 'sp2', 'type': 'num'},
                {'id': '100', 'tokens': 'sp100', 'type': 'num'},
                {'id': '360', 'tokens': 'sp360', 'type': 'num'},
                {'id': '1000', 'tokens': 'sp1000', 'type': 'num'},
                {'id': '90', 'tokens': 'sp90', 'type': 'num'},
                {'id': '180', 'tokens': 'sp180', 'type': 'num'},
                {'id': '10', 'tokens': 'sp10', 'type': 'num'},
                {'id': '12', 'tokens': 'sp12', 'type': 'num'},
                {'id': '3', 'tokens': 'sp3', 'type': 'num'},
                {'id': '30', 'tokens': 'sp30', 'type': 'num'},
                {'id': '0.01', 'tokens': 'sp0.01', 'type': 'num'},
                {'id': 'pi', 'tokens': 'sp_pi', 'type': 'num'}]
implied_multi_ents = [{'id': '1', 'tokens': ['sp1'], 'type': 'num'},
                      {'id': '4', 'tokens': ['sp4'], 'type': 'num'},
                      {'id': '60', 'tokens': ['sp60'], 'type': 'num'},
                      {'id': '2', 'tokens': ['sp2'], 'type': 'num'},
                      {'id': '10', 'tokens': ['sp10'], 'type': 'num'}]
imp_word_dict = {w['word']: w for w in implied_words}
imp_ent_dict = {w['id']: w for w in implied_ents}
imp_multi_ent_dict = {w['id']: w for w in implied_multi_ents}


def pairs_to_utie(pairs, unk_nums, multi=False):
    from collections import OrderedDict
    new_data = []
    for pid, pair in pairs.items():
        x = pair
        infix_expressions = x[1]
        relations = []
        imply_nums = x[-2]
        imply_ns = []
        nums_pos_dict = x[3]
        for n in imply_nums:
            if n == '3.14':
                imply_ns.append('pi')
            else:
                imply_ns.append(n)
        imply_words = [imp_word_dict[i_num] for i_num in imply_ns]
        if multi:
            imply_ents = [imp_multi_ent_dict[i_num] for i_num in imply_ns]
            unk_ents = [{'id': 'x', 'tokens': ['unk1'], 'type': 'unk'},
                        {'id': 'y', 'tokens': ['unk2'], 'type': 'unk'},
                        {'id': 'z', 'tokens': ['unk3'], 'type': 'unk'}]
        else:
            imply_ents = [imp_ent_dict[i_num] for i_num in imply_ns]
            unk_ents = [{'id': 'x', 'tokens': 'unk1', 'type': 'unk'},
                        {'id': 'y', 'tokens': 'unk2', 'type': 'unk'},
                        {'id': 'z', 'tokens': 'unk3', 'type': 'unk'}]
        unk_words = [{'word': 'x', 'id': 'unk1'},
                     {'word': 'y', 'id': 'unk2'},
                     {'word': 'z', 'id': 'unk3'}]
        extend_words = unk_words[0:unk_nums]
        extend_ents = unk_ents[0:unk_nums]
        extend_words.extend(imply_words)
        extend_ents.extend(imply_ents)
        for i, infix_expression in enumerate(infix_expressions):
            infix_exp = []
            for exp in infix_expression:
                if exp == '3.14':
                    infix_exp.append('pi')
                else:
                    infix_exp.append(exp)
            infix_expression = infix_exp
            try:
                relations.extend(infix_to_relation_with_equal(i, len(relations), infix_expression))
            except:
                print('failed to relations: ', infix_expression)
        w_dict = {'”': '"', '“': '"', u'”': '"', u'“': '"', '\u3000': ' ', '\uff1f': ' '}
        if multi:
            sample = {
                'info': {'sid': pid, 'q': x},
                'words': [{'word': w_dict.get(w, w), 'id': str(i)} for i, w in enumerate(x[0])] + extend_words,
                'entities': [{'id': 'N{}'.format(i), 'tokens': nums_pos_dict[v], 'type': 'num'} for i, v in
                             enumerate(nums_pos_dict)] + extend_ents,
                'relations': relations
            }
        else:
            sample = {
                'info': {'sid': pid, 'q': x},
                'words': [{'word': w_dict.get(w, w), 'id': str(i)} for i, w in enumerate(x[0])] + extend_words,
                'entities': [{'id': 'N{}'.format(i), 'tokens': str(v), 'type': 'num'} for i, v in
                             enumerate(x[3])] + extend_ents,
                'relations': relations
            }
        add_significant_number(sample)
        try:
            sample = reid(sample, unordered_types=['+', '*', '='], readable=False)
            sample['relations'] = list(OrderedDict((r['id'], r) for r in sample['relations']).values())
            new_data.append(sample)
        except:
            print('reid falied: ', sample)
    print('{} pairs, {} success to convert to utie'.format(len(pairs), len(new_data)))
    return new_data


def bad_data_handle(s):
    if s['id'] == 'yahoo.answers.20100606182013aaboifr':
        s['equations'] = s['equations'].replace('2*x -11', '2*x - 11')
    if s['id'] == 'yahoo.answers.20080831103207aaxfkfc':
        s['equations'] = s['equations'].replace('3+3/8', '27/8')
    if s['id'] == 'yahoo.answers.20110822104552AA5dVIH':
        s['equations'] = s['equations'].replace('1.4', '(1 + 0.4)')
    if s['id'] == 'yahoo.answers.20110209203154AAE7Sz4':
        s['equations'] = s['equations'].replace('600', '200 * 3')
    if s['id'] == 'yahoo.answers.20070501114420AAd9oYf':
        s['equations'] = s['equations'].replace('/0', '/40')
    if s['id'] == 'yahoo.answers.20071117163907AAzuPdi':
        s['ans'] = '20/3 or 55'
    if s['id'] == 'yahoo.answers.20071117163907AAzuPdi':
        s['ans'] = '1/12'
    if s['id'] == 'yahoo.answers.20080117232720AAfxYrt':
        s['text'] = s['text'].replace('0.3', '. 3')
    if s['id'] == 'yahoo.answers.20080916195022AAQsn1l':
        s['text'] = s['text'].replace('165.000', '165,000')
    return s
