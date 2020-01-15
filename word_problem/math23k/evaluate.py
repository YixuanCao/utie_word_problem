# coding=utf8

from __future__ import print_function, division, unicode_literals
import re
import itertools

import numpy as np

from usage_example.word_problem.math23k.process import to_float
from utie.common import Replacer
from utie.graph_common import remove_ancestors


def sni_result_postprocess(predict):
    all_numbers = {e['id'] for e in predict['entities']}
    sig_numbers = {r['operands'][0] for r in predict['relations'] if r['type'] == 'S'}
    for r in predict['relations']:
        if r['type'] == 'S':
            r['remove'] = True
        for op in r['operands']:
            if op in all_numbers and op not in sig_numbers:
                r['remove'] = True
    filtered_relations = remove_ancestors(predict['relations'])
    if len([r for r in predict['relations'] if r['type'] != 'S']) != len(filtered_relations):
        print(predict['info']['sid'])
    predict['relations'] = filtered_relations


def print_dict(entities):
    _pdict = {}
    entities = sorted(entities, key=lambda x: int(re.search(r'num:w(\d+)|.+', x['id']).groups(0)[0]))
    for i, e in enumerate(entities):
        _pdict[e['id']] = 'N{}'.format(i)
    return _pdict


def get_answer_value(sample, v=False):
    def op_is_entity(op):
        return op in entity_to_value

    def op2value(op):
        if op_is_entity(op):
            return entity_to_value[op]
        return relation_to_value[op]

    def op2p(op):
        if op_is_entity(op):
            return 1.
        return rdict[op]['cum_p']

    def calculate(oper1, oper2, ele):
        if ele == '+':
            return oper1 + oper2
        elif ele == '-':
            return oper1 - oper2
        elif ele == '*':
            return oper1 * oper2
        elif ele == '/':
            if oper2 == 0:
                print('x / 0', sample)
                return -10000
            return oper1 / oper2
        elif ele == '^':
            return oper1 ** oper2
        else:
            print(ele)

    rdict = {r['id']: r for r in sample['relations']}
    entity_to_value = {e['id']: to_float(v) for e, v in zip(sample['entities'][: -1], sample['info']['q'][2])}
    entity_to_value.update({sample['entities'][-2]['id']: 1.})
    entity_to_value.update({sample['entities'][-1]['id']: 3.14})
    relation_to_value = {r['id']: None for r in sample['relations']}
    for r in sample['relations']:
        if r['type'] == '=':
            r['value'] = op2value(r['operands'][0])
            r['cum_p'] = r['prob']
        else:
            op1v, op2v = op2value(r['operands'][0]), op2value(r['operands'][1])
            r['value'] = calculate(op1v, op2v, r['type'])
            relation_to_value[r['id']] = r['value']
            r['cum_p'] = op2p(r['operands'][0]) * op2p(r['operands'][1]) * r['prob']

    if not sample['relations']:
        return None
    # find the root with highest prob
    operands = set()
    for r in sample['relations']:
        operands.update(r['operands'])
    roots = [r for r in sample['relations'] if r['id'] not in operands]
    roots = sorted(roots, key=lambda _x: _x['cum_p'])
    if len(roots) > 1:
        if round(to_float(sample['info']['q'][-1]), 3) != [round(x['value'], 3) for x in roots][-1]:
            pdict = print_dict(sample['entities'])
            replacer = Replacer(pdict)
            if v:
                print(u''.join(sample['info']['q'][0]), u''.join(sample['info']['q'][1]))
                print(sample['info']['q'][-1])
                for r in roots:
                    print(r['cum_p'], r['value'], replacer(r['id']))
                print('')
    return [r['value'] for r in roots]


def try_merge_multi_root(sample, roots, pfc):
    pfc.return_neg = True
    all_results = pfc.predict([sample])[0]
    roots_id = [r['id'] for r in roots]
    combinations = set(itertools.product(roots_id, roots_id))
    higher_root = None
    prob = 0
    for r in all_results['relations']:
        if tuple(r['operands']) in combinations:
            if r['probabilities'][0] < 0.999:
                np.argsort(r['probabilities'])
                print(r)
    pfc.return_neg = False


def get_value_correct(predicted, v=False):
    correct_set = []
    multi = []
    correct_highest = []
    for i, s in enumerate(predicted):
        s['my_answer'] = get_answer_value(s, v=v)
        if s['my_answer'] is None:
            correct_set.append(False)
            correct_highest.append(False)
            multi.append(False)
        else:
            annotate_answer = round(to_float(s['info']['q'][-1]), 3)
            my_answers = [round(x, 3) for x in s['my_answer']]
            bingo_set = annotate_answer in my_answers
            bingo_high = annotate_answer == my_answers[-1]
            multi.append(len(s['my_answer']) > 1)
            correct_set.append(bingo_set)
            correct_highest.append(bingo_high)

    multi = np.asarray(multi)
    correct_set = np.asarray(correct_set)
    output = """
    -------------------------------------------------------
    samples: {}
    equation acc: {}({}
    answer acc: {}({}
    #(multi & correct in set),  #(multi & correct highest) {}, {}
    -------------------------------------------------------
    """.format(len(correct_highest),
               round(sum(correct_highest) / len(correct_highest) * 100, 2), sum(correct_highest),
               round(sum(correct_set) / len(correct_set) * 100, 2), sum(correct_set),
               np.bitwise_and(correct_set, multi).sum(), np.bitwise_and(correct_highest, multi).sum())
    print(output)
    return round(sum(correct_highest) / len(correct_highest) * 100, 2)
