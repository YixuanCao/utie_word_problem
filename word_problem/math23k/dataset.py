# coding=utf8

from __future__ import print_function, division, unicode_literals

from utie.data_label_to_candidate import reid
from utie.dataset import UtieDataset
from usage_example.word_problem.math23k.process import eq_da, infix_to_relation, \
    consistent_expression_and_answer, add_significant_number


class DADataset(UtieDataset):
    def modify_sample_in_getitem(self, l_sample):
        x = l_sample['info']['q']
        infix_expression = x[1]
        relations = []
        if '+' in x[0] or '*' in x[0] or '-' in x[0]:
            infix_expressions = [infix_expression]  # 直接是表达式的，不做处理
        else:
            infix_expressions = list(eq_da(infix_expression, 0.5, False))
        assert isinstance(infix_expressions[0], (list, tuple)),\
            (infix_expressions, type(infix_expressions[0]))
        for one_eq in infix_expressions:
            relations.extend(infix_to_relation(one_eq))
        # success, consistent, ans_v = consistent_expression_and_answer(x[2], infix_expressions[0], x[-1])
        # assert consistent, (ans_v, infix_expressions[0], l_sample)

        w1 = {'word': '1', 'id': 'sp1'}
        e1 = {'id': '1', 'tokens': 'sp1', 'type': 'num'}
        w_pi = {'word': 'π', 'id': 'sp_pi'}
        e_pi = {'id': '3.14', 'tokens': 'sp_pi', 'type': 'num'}

        w_dict = {'”': '"', '“': '"', u'”': '"', u'“': '"', u'NUM': 'n', '\u3000': ' ', '\uff1f': ' '}
        sample = {
            'info': l_sample['info'],
            'words': [{'word': w_dict.get(w, w), 'id': str(i)} for i, w in enumerate(x[0])] + [w1, w_pi],
            'entities': [{'id': 'N{}'.format(i), 'tokens': str(v), 'type': 'num'} for i, v in enumerate(x[3])] + [e1, e_pi],
            'relations': relations
        }
        add_significant_number(sample)
        sample = reid(sample, unordered_types=['+', '*', '='], readable=False)
        return sample
