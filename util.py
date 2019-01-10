#coding=utf-8
"""
@version=1.0
@author:zsh
@file:main.py
@time:2019/1/8 15:00
"""
import torch as t
import time,math,os,pickle,unicodedata,random,re
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
from torch.utils.data import  Dataset
SOS_token = 0
EOS_token = 1
class Lang(object):
    def __init__(self,name):
        self.name=name
        self.word2index = {}
        self.word2count ={}
        self.index2word = {0:'SOS',1:'EOS'}
        self.n_words = 2

    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word]=self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words]=word
            self.n_words+=1
        else:
            self.word2count[word]+=1

    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    def __str__(self):
        return ('name is %s has %d n_words' % (self.name,self.n_words))

def unicode2Ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')

def normalizeString(s):
    s=unicode2Ascii(s.lower().strip())
    s=re.sub(r"(['.!?])",r" ",s)
    s=re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    return s

def readLangs(lang1,lang2,reverse = False):
    print('reading lines')
    filename = 'data/'+lang1+'-'+lang2+'.txt'
    if not os.path.exists('./data/input.pkl'):
        with open(filename,'r') as fr:
            lines = fr.read().strip().split('\n')
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang=Lang(lang2)
        with open('./data/input.pkl','wb') as fw:
            pickle.dump((input_lang,output_lang,pairs),fw)
        return input_lang,output_lang,pairs
    else:
        with open('./data/input.pkl','rb') as fr:
            data=pickle.load(fr)
            input_lang, output_lang, pairs=data[0],data[1],data[-1]
            return input_lang,output_lang,pairs
def filterPair(p,MAX_LENGTH,eng_prefixes):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' '))<MAX_LENGTH and p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    eng_prefixes = ('i am', 'i m', 'he is', 'he s', 'she is', 'she s', 'you are', 'you re', 'we are', 'we re',
                    'they are', 'they re')
    MAX_LENGTH = 10
    return [pair for pair in pairs if filterPair(pair,MAX_LENGTH,eng_prefixes)]

def prepareData(lang1,lang2,reverse=False):
    input_lang,output_lang,pairs = readLangs(lang1,lang2,reverse=reverse)
    print('read %d sentence pairs' % len(pairs))
    print('random', random.choice(pairs))
    pairs = filterPairs(pairs)

    print('after filtered %d sentence pairs' % len(pairs))
    print('random', random.choice(pairs))
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print(input_lang,output_lang)
    return input_lang,output_lang,pairs

def indexFromSentence(lang,sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]
def tensorFromSentence(lang,sentence):
    indexes = indexFromSentence(lang,sentence)
    indexes.append(EOS_token)
    result = t.LongTensor(indexes)
    return result
def tensorFromPair(input_lang,output_lang,pair):
    input_tensor = tensorFromSentence(input_lang,pair[0])
    output_tensor = tensorFromSentence(output_lang,pair[1])
    return input_tensor,output_tensor
def asMinutes(s):
    m = math.floor(s/60)
    s-=m*60
    return '%dm %ds' %(m,s)

def timeSince(since,percent):
    now = time.time()
    s=now-since
    es=s/percent
    rs=es-s
    return '%s (- %s)' %(asMinutes(s),asMinutes(rs))

def test():
    # input_lang, output_lang, pairs=prepareData('eng','fra')
    start = time.time()
    i=0
    while i<100:
       i+=1
       time.sleep(1)
       print(timeSince(start, i/100))

class TextDataset(Dataset):
    def __init__(self,dataload = prepareData,lang = ['eng','fra']):
        self.input_lang,self.output_lang,self.pairs = dataload(lang[0],lang[1],reverse=True)
        self.input_lang_words = self.input_lang.n_words
        self.output_lang_words = self.output_lang.n_words
    def __getitem__(self, index):
        return tensorFromPair(self.input_lang,self.output_lang,self.pairs[index])
    def __len__(self):
        return len(self.pairs)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
if __name__ == '__main__':
    test()
    print('\1')
    # s='''it may be impossible to get a completely error free corpus due to the nature of this kin'''
    # print(s.startswith(('it s','it may be')))