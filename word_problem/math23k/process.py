# coding=utf8
"""
处理 Math23K 的数据
使用到了 math_pade_da-master 的处理代码
"""
from __future__ import unicode_literals, division, print_function
import re
import copy
import decimal

# sys.path.append('/Users/caoyixuan/Downloads/实验数据/math_pade_da-master')
from src.pre_data import load_raw_data, transfer_num, check_bracket, allocation, exchange


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


def to_float(s):
    """把string变成float
    可以处理的形式： (1)/3  (1/(3))   1/3  30%
    round 是因为计算不精确 1.8 / 100 = 0.018000000000000002
    """
    r = r'\d+\.?\d*'
    reg = re.compile(r'({r})\(\(?({r})\)?/\(?({r})\)?\)'.format(r=r))  # 1 3/4 = 1+3/4
    m = reg.match(s)
    if m:
        a, b, c = map(to_float, m.groups())
        return a + b / c

    if '(' in s:
        s = re.sub(pattern=r'(\d+)', repl=r'\g<1>.0', string=s, count=1)
        return eval(s)
    s = s.strip('()')
    if '/' in s:
        f1, f2 = s.split('/')
        return float(decimal.Decimal(f1) / decimal.Decimal(f2))
    if s.endswith('%'):
        return float(s[:-1]) / 100.
    if s.endswith('percent'):
        return float(s[:-7]) / 100.
    return float(s)


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


def postfix_to_tree(postfix_expression):
    # 利用后缀表达式计算结果
    stack = []
    i = 0
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


def consistent_expression_and_answer(value_list, infix_expression, ans_str):
    # value_list: [1, 2.5, ..] 句子中出现的值，按出现前后排序
    # infix_expression: 用符号表示的 infix_expression
    # return Success, consistent
    try:
        value_dict = {'N{}'.format(i): to_float(v) for i, v in enumerate(value_list)}  # Ni-> float
        ans = to_float(ans_str)
        replace_dict = {'[': '(', ']': ')'}
        infix_expression = [replace_dict.get(z, z) for z in infix_expression]
        infix = [value_dict.get(z, z) for z in infix_expression]
        v = eval_postfix(infix_to_postfix(infix))
        return True, round(v, 5) == round(ans, 5), (ans, v)
    except:
        print('fail to float {} on {}'.format(ans_str, infix_expression))
        return False, None, None


def infix_to_relation(infix_expression):
    # 中括号变成小括号
    replace_dict = {'[': '(', ']': ')'}
    infix_expression = [replace_dict.get(z, z) for z in infix_expression]
    tree = postfix_to_tree(infix_to_postfix(infix_expression))
    if not isinstance(tree, Tree):
        # print('not tree', I, tree, x)
        relations = [{'id': 'eq', 'type': '=', 'operands': [tree]}]  # 特殊对待 `把1/3写成小数` 这样的题目
    else:
        relations = [{'id': n[0], 'operands': [n[1], n[2]], 'type': n[-1]} for n in tree.traverse()]
    return relations


def add_significant_number(sample):
    used_entities = set()
    for r in sample['relations']:
        used_entities.update(r['operands'])
    significant_number_indicator = []
    for e in sample['entities']:
        if e['id'] in used_entities:
            significant_number_indicator.append({'id': 'SNI-{}'.format(e["id"]), 'operands': [e['id']], 'type': 'S'})
    sample['relations'].extend(significant_number_indicator)


char_num = re.compile(r'[Ne]+\d+')


def check_num_op(sample):
    valid = True
    fail_info = ''
    for r in sample['relations']:
        for op in r['operands']:
            if not (char_num.match(op) or op == '1' or op == '3.14'):
                valid = False
                fail_info = 'invalid op: {} in {}'.format(op, r)
                break
    return valid, fail_info


def one_step_da(eq_original, rate, english=False):
    eq_original = copy.deepcopy(eq_original)
    equations = set()
    for i in range(3):
        eq_original = check_bracket(eq_original, english)
        equations.add(tuple(eq_original))

        temp_out = check_bracket(exchange(eq_original, rate), english)

        temp_out_a = check_bracket(allocation(eq_original, rate), english)
        if temp_out_a != eq_original:
            equations.add(tuple(temp_out_a))

        if temp_out != eq_original:
            equations.add(tuple(temp_out))

        if temp_out_a != eq_original and temp_out != eq_original:
            temp_out_a = check_bracket(allocation(temp_out, rate), english)
            if temp_out_a != temp_out:
                equations.add(tuple(temp_out_a))
    return equations


def eq_da(eq_original, rate, english=False):
    eqs = set()
    eqs.add(tuple(eq_original))
    for i in range(2):
        new = set()
        for eq in eqs:
            eq = list(eq)
            new.update(one_step_da(eq, rate, english))
        eqs.update(new)
    eqs = list(eqs)
    eqs.remove(tuple(eq_original))
    return [eq_original] + eqs



def nfold(data, n, test_idx):
    assert 0 <= test_idx < n
    fold_size = int(len(data) / float(n))
    train = []
    test = []
    for i in range(n):
        s = i * fold_size
        e = (i + 1) * fold_size
        if i == n - 1:
            e = len(data)
        one_fold = data[s: e]
        if i == test_idx:
            test = one_fold
        else:
            train.extend(one_fold)
    return train, test
