# coding=utf8

"""
Annotating Derivations: A New Evaluation Strategy and Dataset for Algebra Word Problems
这篇文章发布了几个数据集  Alg514, DRAW1K，都是有方程组的数据，并且进行了 Align
先看EACL文件夹中的 readme.txt 文件了解格式信息


delphin: 831,  500 60% 成功
draw 1k : 1000, 905 90% 成功, 最高 59.5
kushman: 514, 430 83% 成功, 最高 83
"""

from word_problem.dolphin.process import EqParse, to_float, EnglishToNumber, SpecialNums


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
        for v in align_info:
            token_idx, value = align_info[v]
            value = round(value, 5)
            token = tokens[token_idx]
            try:
                token = EnglishToNumber.handle(SpecialNums.handle(token.lower()))
                if ' ' in token:
                    token = token.split(' ')[0]
                if round(to_float(token), 5) != round(float(value), 5):
                    print(token, value, problem['sQuestion'])
                    return False
            except:
                print(token)  #, value, problem['sQuestion'])
                return False
        if problem['Equiv']:
            print(problem['Equiv'], problem['sQuestion'])
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
