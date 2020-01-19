# coding=utf8

"""
Annotating Derivations: A New Evaluation Strategy and Dataset for Algebra Word Problems
这篇文章发布了几个数据集  Alg514, DRAW1K，都是有方程组的数据，并且进行了 Align
先看EACL文件夹中的 readme.txt 文件了解格式信息


delphin: 831,  500 60% 成功
draw 1k : 1000, 905 90% 成功, 最高 59.5, 主要问题在 dime, nickle, 可以考虑把他们也作为数字
kushman (alg514): 514, 430 83% 成功, 最高 83
"""

from utie.common import ReSegment
from word_problem.dolphin.process import EqParse, QuestionAnswerParser, extract_numbers, parse_answer_equations, pairs_to_utie
from word_problem.math23k.process import to_float
from word_problem.dolphin.text_num import EnglishToNumber, SpecialNums


def parse_align(problem):
    """
     {
          "sQuestion": "A factory makes three-legged stools and four-legged stools . They use the same kind of seats and legs for each . The only difference is the number of holes they bore in the seats for the legs . The company has 161 seats and 566 legs .",
          "lSolutions": [
           83.0,
           78.0
          ],
          "Template": [
           "a * m + b * n = c",
           "m + n = d"
          ],
          "lEquations": [
           "student+general=161",
           "3*student+4*general=566"
          ],
          "iIndex": 327651,
          "Alignment": [
           {
            "coeff": "a",
            "SentenceId": 0,
            "Value": 3.0,
            "TokenId": 3
           },
           {
            "coeff": "b",
            "SentenceId": 0,
            "Value": 4.0,
            "TokenId": 6
           },
           {
            "coeff": "c",
            "SentenceId": 3,
            "Value": 566.0,
            "TokenId": 6
           },
           {
            "coeff": "d",
            "SentenceId": 3,
            "Value": 161.0,
            "TokenId": 3
           }
          ],
          "Equiv": []
     },
    :param problem:
    :return:
    """
    def check():
        """检查数字和 token 对齐"""
        if problem['Equiv']:
            print('more than one match: ', problem['Equiv'], problem['sQuestion'])
            return False
        for v in align_info:
            token_idx, value = align_info[v]
            value = round(value, 5)
            token = tokens[token_idx]
            try:
                token = EnglishToNumber.handle(SpecialNums.handle(token.lower(), no_percent=True))
                if ' ' in token:
                    token = token.split(' ')[0]
                if round(to_float(token), 5) != round(float(value), 5):
                    print('token not equal to value: ', problem['Alignment'], token, value, problem['sQuestion'])
                    return False
            except:
                print('align not equal in text: ', problem['Alignment'], token, value, problem['sQuestion'])  #, value, problem['sQuestion'])
                return False
        return True

    sent_id_to_offset = {0: 0}
    sent_i = 1
    tokens = problem['sQuestion'].split()
    for i, token in enumerate(tokens):
        if token in '?.!':
            sent_id_to_offset[sent_i] = i + 1
            sent_i += 1
    templates = [EqParse.parse(t) for t in problem['Template']]
    align_info = {}  # 变量名: [token id, value]
    for align in problem['Alignment']:
        align_info[align['coeff']] = [sent_id_to_offset[align['SentenceId']] + align['TokenId'], align['Value']]
    return check()


class DQuestionAnswerParser(QuestionAnswerParser):

    def __init__(self, qa):
        super(DQuestionAnswerParser, self).__init__(qa)

    @staticmethod
    def add_text_implied_nums(text):
        implied_list = {
                        # ' odd ': ' odd 2 ',
                        # ' even ': ' even 2 ',
                        ' couple ': ' couple 2 ',
                        ' tens ': ' tens 10 ',
                        ' a quarter ': ' 1/4 ',
                        ' quarters ': ' quarters 0.25 ',
                        ' a third ': ' 1/3 ',
                        ' nickles ': ' nickles 0.05 ',
                        ' nickels ': ' nickels 0.05 ',
                        ' dimes ': ' dimes 0.1 ',
                        ' cent ': ' cent 0.01 ',
                        '-cent': '-cent 0.01',
                        'cents': 'cents 0.01',
                        'perimeter': 'perimeter 2',
                        'square': 'square 2',
                        'the opposite of': 'the opposite -1 of',
                        'pure fruit juice': 'pure 100 fruit juice',
                        'pennys': 'pennys 0.01',
                        'supplementary': 'supplementary 180',
                        'equal amount of': 'equal 2 amount of',
                        '%': '% 0.01',
                        'percent': 'percent 0.01'
                        }
        for k, v in implied_list.items():
            if k in text:
                text = text.replace(k, v, 1)
        implied_nums = {'1'}
        if 'digit' in text:
            implied_nums.add('10')
        if 'heads' in text and 'feet' in text:
            implied_nums.add('4')
            implied_nums.add('2')
        if ' even ' in text or ' odd ' in text:
            implied_nums.add('2')

        return text, implied_nums

    def parse(self, no_percent=True):
        qa = self.qa
        text, implied_nums = self.add_text_implied_nums(qa['sQuestion'].lower())
        new_text, nums = extract_numbers(text, no_percent)
        s, unknowns, eqs = parse_answer_equations(qa['lEquations'], {x[0] for x in nums}, no_percent)
        numbers_in_eqs = set()
        for eq in eqs:
            if not ('>' in eq or '<' in eq):
                numbers_in_eqs.update([ele for ele in eq if EqParse.ele_type(ele) == 'num'])
        numbers_in_text = {x[0] for x in nums}
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
                pair = self.transfer_num(new_text, eqs, numbers_in_eqs, numbers_in_text, unknowns, no_percent)
                new_segs, eq_segs, nums, num_pos = pair
                return new_segs, eq_segs, nums, num_pos, list(implied_nums), qa['lSolutions']


def draw_data_to_utie(data_path):
    import json
    draw_data = json.load(open(data_path))
    pairs = draw_data_to_pairs(draw_data)
    draw_utie = pairs_to_utie(pairs, 2)
    return draw_utie


def draw_data_to_pairs(draw_data):
    pairs = {}
    for i, s in enumerate(draw_data):
        if s['Equiv']:
            continue
        s = bad_draw_handle(s)
        try:
            qa_p = DQuestionAnswerParser(s)
            pair = qa_p.gen_pair(True)
            if pair[2]:
                pairs[s['iIndex']] = pair
        except:
            print('Data to pair failed: ', s)
            pass
    print('{} samples, {} success to convert to pairs'.format(len(draw_data), len(pairs)))
    return pairs


def bad_draw_handle(s):
    zero_indexs = [676077, 517110, 838940, 722565, 929086, 1042, 839861]
    if s['iIndex'] in zero_indexs:
        new_eqs = []
        for eq in s['lEquations']:
            if '=0' in eq:
                eq = eq.replace('=0', '').replace('-', '=')
                new_eqs.append(eq)
            elif '= 0' in eq:
                eq = eq.replace('= 0', '').replace('-', '=')
                new_eqs.append(eq)
            else:
                new_eqs.append(eq)
        s['lEquations'] = new_eqs
    if s['iIndex'] == 174785:
        s['sQuestion'] = s['sQuestion'].replace('25 ,000', '25,000')
    if s['iIndex'] == 369466:
        s['lEquations'] = [s['lEquations'][0].replace('0.16666', '0.16667')]
    if s['iIndex'] == 928210:
        s['lEquations'][0] = 'x*(1-0.01*20)-80=80*0.01*30'
    if s['iIndex'] == 189327:
        s['lEquations'] = ['40 / 2 * x + 40 / 2 * y = 1.05 *( 40 )', 'y=(x-0.3)']
    if s['iIndex'] == 751866:
        s['sQuestion'] = s['sQuestion'].replace('5c and 10c', '0.05 and 0.10')
    return s


def main():
    import json
    draw_raw = json.load(open('data/EACL/draw.json'))
    sum(parse_align(q) for q in draw_raw)


if __name__ == '__main__':
    main()
