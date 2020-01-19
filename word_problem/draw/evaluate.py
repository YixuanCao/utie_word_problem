# coding=utf8

import argparse
from utie.test_pipeline import TestPipeline
import regex as re


def filted_data(predict_samples):
    """
    过滤掉没有‘=’的sample
    """
    avail_samples = []
    for i in predict_samples:
        for rel in i['relations']:
            if rel['type'] == '=':
                avail_samples.append(i)
                break
    return avail_samples


def solve_equations(equations, precision):
    """解方程组, 未知数最多为3个，且需在x,y,z中
    :param equations: 方程构成的方程组列表，需将方程变换成等式右边为0的情况，且只保留左边， 如解 x-1=y；x+y=3 输入[x-1-y, x+y-3]
    :param precision: 解的精度
    :return: 方程的解, 唯一解时按[x,y,z]的先后顺序排列，省略未知数信息，有多个解时会保留对应的未知数信息
    """
    from sympy import solve, Symbol
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    try:
        equations = [eval(eq) for eq in equations]
        answer = solve(equations)
    except:
        answer = []
        print('wrong format of equations: ', equations)
    try:
        if isinstance(answer, list):
            new_answer = []
            for ans in answer:
                for k, v in ans.items():
                    new_answer.append((k, round(float(v), precision)))
            answer = new_answer
        else:
            #             real_ans = {(k, round(float(v), 3)) for k, v in real_ans.items()}
            answer = [round(float(v), precision) for k, v in answer.items()]
    except:
        print('wrong format of answer: ', answer)
    return answer


def get_real_ans(origin_eqs, values, precision):
    """
    :param origin_eqs: 对应标注数据中的公式 [['x', '=', 'y', '-', 'N1'],['N2', '*', 'x', '-', 'N3', '*', 'y', '=', 'N4']]
    :param values: 文本中nums对应的values列表 ['1', '4', '2', '5', '-11'],
    :param precision: 答案保留的精度
    :return: 公式对应的解的集合，有多个解时会保留对应的未知数信息, 以及原方程的变形， x-1=y --> x-1-y
    """
    new_eqs = []
    for eq in origin_eqs:
        new_eq = []
        for e in eq:
            if e[0] == 'N':
                e = values[int(e[1:])]
                if '/' in e:
                    new_eq.append('({})'.format(e))
                else:
                    new_eq.append(e)
            else:
                new_eq.append(e)
        new_eq = ''.join(new_eq).split('=')
        new_eq = '{}-({})'.format(new_eq[0], new_eq[1])
        new_eqs.append(new_eq)
    real_ans = solve_equations(new_eqs, precision)
    return real_ans, new_eqs


def predict_relation_to_eqs(predict_sample):

    def replace_eq(op, rel2eq):
        # 从长找到短
        for rel in rel2eq:
            if rel[0] in op:
                op = op.replace(rel[0], rel[1])
                replace_eq(op, rel2eq)
        else:
            return op

    def replace_v(op, wid2value):
        for wid, v in wid2value.items():
            if wid in op:
                op = op.replace(wid, v)
                replace_v(op, wid2value)
        else:
            return op

    def op_id_to_term(op):
        if op in ent2v:
            return ent2v[op]
        if op in rel2eq:
            return rel2eq[op]
        print(op)
        raise ValueError()

    rel2eq = {}
    _, origin_eqs, values, _, _, origin_ans = predict_sample['info']['q']
    idx = 0
    wid2value = {}
    ent2v = {}
    eqs = []
    eq_corresponding_rids = []
    for i, w in enumerate(predict_sample['words']):
        if w['word'] == 'n' and not predict_sample['words'][i + 1]['word'].startswith('##'):
            wid2value[w['id']] = values[idx]
            idx += 1
        else:
            wid2value[w['id']] = w['word']
    for e in predict_sample['entities']:
        if e['tokens'] in wid2value:
            if '/' in wid2value[e['tokens']]:
                ent2v[e['id']] = '({})'.format(wid2value[e['tokens']])
            else:
                ent2v[e['id']] = wid2value[e['tokens']]
    for rel in predict_sample['relations']:
        op0 = op_id_to_term(rel['operands'][0])
        op1 = op_id_to_term(rel['operands'][1])
        if rel['type'] != '=':
            rel2eq[rel['id']] = '({}{}{})'.format(op0, rel['type'], op1)
        if rel['type'] == '=':
            eq = '{}-{}'.format(op0, op1)
            eqs.append(eq)
            eq_corresponding_rids.append(rel['id'])
    rel2eq = sorted(rel2eq.items(), key=lambda x: len(x[0]), reverse=True)
    # predict_eqs = []
    # for eq in eqs:
    #     eq = replace_eq(eq, rel2eq)
    #     eq = replace_v(eq, ent2v)
    #     predict_eqs.append(eq)
    predict_eqs = eqs
    return predict_eqs, eq_corresponding_rids


def process_dolphin_ans(origin_ans, precision):
    from word_problem.math23k.process import to_float
    unks = ['x', 'y', 'z']
    if '|' in origin_ans:
        origin_ans = origin_ans.split('|')[0]
    if 'or' in origin_ans:
        origin_ans = origin_ans.split('or')
    try:
        if isinstance(origin_ans, list):
            # 答案有多个时会生成一个list
            new_origin_ans = []
            for ans in origin_ans:
                ans = ans.strip('{} ?$').split(';')
                if isinstance(ans, str):
                    ans = [ans]
                for i, a in enumerate(ans):
                    new_origin_ans.append((unks[i], round(to_float(re.sub('km\^2|/?[a-zA-Z]|\s', '', a)), precision)))
            origin_ans = set(new_origin_ans)
        else:
            origin_ans = origin_ans.strip('{} ?$').split(';')
            #             origin_ans = {(unks[i], round(to_float(re.sub('km\^2|/?[a-zA-Z]|\s', '', a)), 3)) for i, a in enumerate(origin_ans)}
            origin_ans = {round(to_float(re.sub('km\^2|/?[a-zA-Z]|\s', '', a)), precision) for i, a in enumerate(origin_ans)}
    except:
        print('wrong format of origin answer:', origin_ans)
    return origin_ans



def evaluate(log_path, data_path, precision=3, dolphin=False):
    tppl = TestPipeline(log_path, data_path=data_path, config_path=None, use_best=True)
    performance, indicator, wrong_sids, correct_sids = tppl.evaluate()
    predict_samples = tppl.predicted
    sample_num = len(predict_samples)
    avail_results = filted_data(predict_samples)
    right_num = 0
    right_num_fixed = 0
    for predict_sample in avail_results:
        predict_eqs, _ = predict_relation_to_eqs(predict_sample)
        _, origin_eqs, values, _, _, origin_ans = predict_sample['info']['q']
        if dolphin:
            origin_ans = process_dolphin_ans(origin_ans, precision)
        else:
            origin_ans = [round(float(v), precision) for v in origin_ans]
        real_ans, origin_equations = get_real_ans(origin_eqs, values, precision)
        predict_ans = solve_equations(predict_eqs, precision)
        if set(predict_ans) == set(real_ans):
            right_num_fixed += 1
        else:
            print('predict_answer not equal to real_answer: ', predict_eqs, predict_ans, origin_equations, real_ans)
        if predict_ans and set(predict_ans).issubset(origin_ans):
            right_num += 1
        else:
            print('predict_answer not equal to origin_answer: ', predict_eqs, predict_ans, origin_equations, origin_ans)
    output = """
    -------------------------------------------------------
    test_performance: {}
    answer compared with origin acc: {}({})
    answer compared with real acc: {}({})
    -------------------------------------------------------
    """.format(performance,
               round(float(right_num) / float(sample_num) * 100, 2), right_num,
               round(float(right_num_fixed) / float(sample_num) * 100, 2), right_num_fixed
               )
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path')
    parser.add_argument('data_path')
    parser.add_argument('precision')
    parser.add_argument('dolphin')
    parser = parser.parse_args()
    evaluate(parser.log_path, parser.data_path, parser.precision, parser.dolphin)
