#encoding=utf-8
import re,collections

#把语料中的单词都抽取出来，变成小写a-z，去掉特殊符号
def words(text):
    return  re.findall('[a-z]+',text.lower())
#
def  train(features):
    model=collections.defaultdict(lambda: 1)#新读取的词都为1，返回model为一个数组
    for f in features:
        model[f] = model[f]+1       #统计文本中每个出现的单词的频率
    return model
#26个小写字母
letter='abcdefghijklmnopqrstuvwxyz'
#读取文本。read，words，这句话就是把大写的字母全部换成小写，有空格或者转行的地方判断为一个新词汇
NWORDS=train(words(open('wor.txt').read()))


#编辑距离，经过增改删cud，交换字母,替换字母的操作从一个词变成另一个词。edits1是编辑距离等于1
def editsl(word):
    n=len(word)

    return set([word[0:i]+word[i+1:]  for i in range(n)]+
               [word[0:i]+word[i+1]+word[i]+word[i+2:]  for i in range(n-1)]+
               [word[0:i]+c+word[i+1:] for i in range(n) for c in letter]+
               [word[0:i]+c+word[i:] for i in range(n+1) for c in letter])
#编辑距离等于2（在编辑距离为1的基础上再加一次）
def known_edits2(word):
    return set(e2 for e1 in editsl(word)
               for e2 in editsl(e1) if e2 in NWORDS)

def known(words):
    return set(w for w in words if w in NWORDS)
#如果known（set）为非空，candidates就会选这个集合。按照次序来，如果为非空编辑距离为0先选他，然后再找1，
#找不到再找2
def correct(word):
    candidates=known([word])  or  known(editsl(word)) or known_edits2(word) or [word]
    return max(candidates,key=lambda w:NWORDS[w])
# a=[[2,4],
#    [5,7],
#    [32,65]
# ]
# print(a[0:2])

#import torch
#print(torch.cuda.is_available())
print(correct('anacon'))


