# coding=utf8

import argparse
from utie.test_pipeline import TestPipeline
import itertools
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
    Symbol('x y z')
    try:
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
        answer = []
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


def get_real_ans_multi(origin_eqs, values, precision):
    """
    :param origin_eqs: 对应标注数据中的公式 [['x', '=', 'y', '-', 'N1'],['N2', '*', 'x', '-', 'N3', '*', 'y', '=', 'N4']]
    :param values: 文本中nums对应的values列表 ['1', '4', '2', '5', '-11'],
    :param precision: 答案保留的精度
    :return: 公式对应的解的集合，有多个解时会保留对应的未知数信息, 以及原方程的变形， x-1=y --> x-1-y
    """
    new_eqs = []
    idx = 0
    value_idx_dict = {}
    for v in values:
        if str(round(to_float(v), 5)) not in value_idx_dict.values():
            value_idx_dict[idx] = str(round(to_float(v), 5))
            idx += 1
    for eq in origin_eqs:
        new_eq = []
        for e in eq:
            if e[0] == 'N':
                e = value_idx_dict[int(e[1:])]
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


def predict_relation_to_eqs_multi(predict_sample):

    def op_id_to_term(op):
        if op in ent2v:
            return ent2v[op]
        if op in rel2eq:
            return rel2eq[op]
        print(op)
        raise ValueError()

    rel2eq = {}
    _, origin_eqs, _, values_dict, _, origin_ans = predict_sample['info']['q']
    wid_value_dict = {}
    idx_list = []
    for v in values_dict:
        for wid in values_dict[v]:
            wid_value_dict[wid] = v
            idx_list.append(int(wid))
    idx_list.sort()
    wid2value = {}
    ent2v = {}
    eqs = []
    eq_corresponding_rids = []
    eq2prob_dict = {}
    idx_num = 0
    for i, w in enumerate(predict_sample['words']):
        if w['word'] == 'n' and idx_num < len(idx_list) and not (predict_sample['words'][i + 1]['word'].startswith('##') or
                                                                  predict_sample['words'][i + 1]['word'].startswith("'")):
            wid2value[w['id']] = wid_value_dict[str(idx_list[idx_num])]
            idx_num += 1
        else:
            wid2value[w['id']] = w['word']
    for e in predict_sample['entities']:
        if e['tokens'][0] in wid2value:
            if '/' in wid2value[e['tokens'][0]]:
                ent2v[e['id']] = '({})'.format(wid2value[e['tokens'][0]])
            else:
                ent2v[e['id']] = wid2value[e['tokens'][0]]
    for rel in predict_sample['relations']:
        op0 = op_id_to_term(rel['operands'][0])
        op1 = op_id_to_term(rel['operands'][1])
        if rel['type'] != '=':
            rel2eq[rel['id']] = '({}{}{})'.format(op0, rel['type'], op1)
        if rel['type'] == '=':
            eq = '{}-{}'.format(op0, op1)
            eqs.append(eq)
            eq2prob_dict[eq] = rel['cum_p']
            eq_corresponding_rids.append(rel['id'])
    predict_eqs = eqs
    return predict_eqs, eq_corresponding_rids, eq2prob_dict


def to_float(s):
    """把string变成float
    可以处理的形式： (1)/3  (1/(3))   1/3  30%
    round 是因为计算不精确 1.8 / 100 = 0.018000000000000002
    """
    import decimal
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


def process_dolphin_ans(origin_ans, precision):
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
            origin_ans = new_origin_ans
        else:
            origin_ans = origin_ans.strip('{} ?$').split(';')
            #             origin_ans = {(unks[i], round(to_float(re.sub('km\^2|/?[a-zA-Z]|\s', '', a)), 3)) for i, a in enumerate(origin_ans)}
            origin_ans = [round(to_float(re.sub('km\^2|/?[a-zA-Z]|\s', '', a)), precision) for i, a in enumerate(origin_ans)]
    except:
        print('wrong format of origin answer:', origin_ans)
    return origin_ans


def include_origin_eqs(predict_eqs, origin_eqs_length, origin_ans, precision):
    """通过组合预测出的所有方程来找出能算出正确答案的组合
    :param predict_eqs:
    :param origin_eqs_length:
    :param origin_ans:
    :param precision:
    :return:
    """
    combs = itertools.combinations(predict_eqs, origin_eqs_length)
    for comb in combs:
        comb_eqs_ans = solve_equations(comb, precision)
        if set(comb_eqs_ans) == set(origin_ans):
            return comb, comb_eqs_ans
    return [], []


def cumulative_prob(sample):
    def op_is_entity(op):
        return op not in relation_dict

    def op2p(op):
        if op_is_entity(op):
            return 1.
        return relation_dict[op]['cum_p']

    relation_dict = {r['id']: r for r in sample['relations']}
    for r in sample['relations']:
        p = r['prob']
        for op in r['operands']:
            p = p * op2p(op)
        r['cum_p'] = p
    return sample


def get_predict_unks_num(predict_eqs):
    """获得预测出的所有公式中未知数的个数，来确定最终的方程组中方程的个数
    :param predict_eqs:
    :return:
    """
    unks_num = 0
    for eq in predict_eqs:
        num = 0
        if 'x' in eq:
            num += 1
        if 'y' in eq:
            num += 1
        if 'z' in eq:
            num += 1
        if unks_num < num:
            unks_num = num
    return unks_num


def get_predict_eq_comb(predict_eqs, unks_num, eq2prob_dict, precision):
    """找出预测出的所有公式中概率最大的公式组合
    :param predict_eqs:
    :param unks_num:
    :param eq2prob_dict:
    :param precision:
    :return:
    """
    combs = itertools.combinations(predict_eqs, unks_num)
    max_prob = 0
    max_comb = ()
    max_comb_ans = []
    for comb in combs:
        answer = solve_equations(comb, precision)
        if answer:
            comb_prob = 1.0
            for eq in comb:
                comb_prob = eq2prob_dict[eq] * comb_prob
            if comb_prob > max_prob:
                max_prob = comb_prob
                max_comb = comb
                max_comb_ans = answer
    return max_comb, max_comb_ans, max_prob


def evaluate(log_path, data_path, precision=3, dolphin=False, use_best=False):
    """
    :param log_path: 训练模型时生成的日志文件存放路径
    :param data_path: 用来评估的数据集路径
    :param precision: 判断答案是否相等时保留的精度
    :param dolphin: 用来评估的是dolphin数据的话为True
    :param use_best: 是否用最好的模型
    :return:
    """
    tppl = TestPipeline(log_path, data_path=data_path, config_path=None, use_best=use_best)
    performance, indicator, wrong_sids, correct_sids = tppl.evaluate()
    predict_samples = tppl.predicted
    def remove_S(samples):
        for sample in samples:
            rels = [rel for rel in sample['relations'] if rel['type'] != 'S']
            sample['relations'] = rels
        return samples
    predict_samples = remove_S(predict_samples)
    sample_num = len(predict_samples)
    avail_results = filted_data(predict_samples)
    right_num = 0
    right_num_fixed = 0
    right_comb_num = 0
    include_num = 0
    predict_eqs_num = 0
    for predict_sample in avail_results:
        predict_sample = cumulative_prob(predict_sample)
        try:
            predict_eqs, _, eq2prob_dict = predict_relation_to_eqs_multi(predict_sample)
            predict_eqs_num += len(predict_eqs)
            # predict_eqs, _ = predict_relation_to_eqs(predict_sample)
            unks_num = get_predict_unks_num(predict_eqs)
            eq_comb, comb_ans, comb_cum_p = get_predict_eq_comb(predict_eqs, unks_num, eq2prob_dict, precision)
        except:
            print('relation to eq failed:', predict_sample)
            continue
        _, origin_eqs, values, _, _, origin_ans = predict_sample['info']['q']
        if dolphin:
            origin_ans = process_dolphin_ans(origin_ans, precision)
        else:
            origin_ans = [round(float(v), precision) for v in origin_ans]
        # real_ans, origin_equations = get_real_ans(origin_eqs, values, precision)
        real_ans, origin_equations = get_real_ans_multi(origin_eqs, values, precision)
        predict_ans = solve_equations(predict_eqs, precision)
        comb, comb_eqs_ans = include_origin_eqs(predict_eqs, len(origin_equations), origin_ans, precision)
        if comb:
            include_num += 1
            if len(predict_eqs) > len(origin_equations):
                print('predict_eqs include origin_eqs:', predict_eqs, comb, comb_eqs_ans, origin_equations, origin_ans, real_ans)
        if comb_ans:
            if set(comb_ans) == set(real_ans):
                right_comb_num += 1
                print('predict_comb equal to origin_eqs:', predict_eqs, eq_comb, comb_cum_p, comb_ans, origin_equations, origin_ans, real_ans)
        if real_ans and set(predict_ans) == set(real_ans):
            right_num_fixed += 1
            # print(predict_sample)
            # print('predict_answer equal to real_answer:', predict_eqs, predict_ans, origin_equations, origin_ans, real_ans)
        # else:
        #     print('predict_answer not equal to real_answer: ', predict_eqs, predict_ans, origin_equations, origin_ans, real_ans)
        if dolphin:
            if predict_ans and set(predict_ans).issubset(set(origin_ans)):
                right_num += 1
            # else:
            #     print('predict_answer not equal to origin_answer: ', predict_eqs, predict_ans, origin_equations, origin_ans, real_ans)
        else:
            if predict_ans and set(predict_ans) == set(origin_ans):
                right_num += 1
            else:
                print('predict_answer not equal to origin_answer: ', predict_eqs, predict_ans, origin_equations, origin_ans, real_ans)
    output = """
    -------------------------------------------------------
    test_performance: {}
    predict equations num: {}
    predict comb acc: {}({})
    answer compared with origin acc: {}({})
    answer compared with real acc: {}({})
    predict equations included origin equations acc: {}({})
    -------------------------------------------------------
    """.format(performance, predict_eqs_num,
               round(float(right_comb_num) / float(sample_num) * 100, 2), right_comb_num,
               round(float(right_num) / float(sample_num) * 100, 2), right_num,
               round(float(right_num_fixed) / float(sample_num) * 100, 2), right_num_fixed,
               round(float(include_num) / float(sample_num) * 100, 2), include_num,
               )
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path')
    parser.add_argument('data_path')
    parser.add_argument('precision')
    parser.add_argument('dolphin')
    parser.add_argument('use_best')
    parser = parser.parse_args()
    evaluate(parser.log_path, parser.data_path, parser.precision, parser.dolphin, parser.use_best)
