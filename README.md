# Solving Math Problem Using UTIE

Python3 

To process Math23K, we need another repo: `git clone https://github.com/ShichaoSun/math_seq2tree.git` 

and you should 

```python
import sys
sys.path.append('path/to/math_seq2tree')
sys.path.append('path/to/UTIE')
```

## Data

in `data` folder

## get started

open jupyter

###  `math23k.ipynb`

- 生成数据标注，把句子中出现的重复实体，在公式中出现的，进行消歧
  不用做这一步，有一些句子，公式中一个数字对应了文本中多个数字，找到这些句子并且扔进 Foundry 进行标注

- 使用标注结果，把答案中的对应关系赋值给 pairs
  需要用Foundry导出的答案(对应关系) 来修改 Math23K 的结果 (Foundey数据在 data 文件夹)

- 解析公式，变成 relations
  主要步骤  



### `dolphin18k.ipynp`  

## Tips

```python
%load_ext autoreload
%autoreload 2
```

放在 jupyter 开头可以让 jupyter 自动重新加载 import 的东西，这样的话， from a import b，在外面修改了 b 之后，b会自动重新加载，再用b的时候使用的最新的 b

