#coding=utf-8
"""
@version=1.0
@author:zsh
@file:evaulate.py
@time:2019/1/10 20:09
"""
import random
import torch as t
from torch.autograd import  Variable
from util import TextDataset
from S_LSTM import Encoder,Decoder
import matplotlib.pyplot as plt

MAX_LENGTH=10
use_attn = False
SOS_token = 0
EOS_token = 1
use_cuda = t.cuda.is_available()
lang_dataset = TextDataset()
print('*'*10)

def convert2Cuda(data):
    return data.cuda() if t.cuda.is_available() else data

def evaluate(encoder,decoder,in_lang,max_length = MAX_LENGTH):
    in_lang=convert2Cuda(in_lang)

    in_variable = Variable(in_lang).unsqueeze(0)
    in_length = in_variable.size(1)
    encoder_hidden=encoder.initHidden()
    encoder_outputs=Variable(t.zeros(max_length,encoder.hidden_size))
    encoder_outputs=convert2Cuda(encoder_outputs)


    for ei in range(in_lang):
        encoder_output,encoder_hidden = encoder(in_variable[:,ei],encoder_hidden)
        encoder_outputs=encoder_output[0][0]
    decoder_input = Variable(t.LongTensor([[SOS_token]]))
    decoder_input=convert2Cuda(decoder_input)
    decoder_hidden=encoder_hidden

    decoder_words = []
    for di in range(max_length):
        decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden)
        topv,topi = decoder_output.data.topk(1)
        ni = topi[0,0]
        if ni==EOS_token:
            decoder_words.append('<EOS>')
            break
        else:
            decoder_words.append(lang_dataset.output_lang.index2word[ni])
        decoder_input=Variable(t.LongTensor([[ni]]))
        decoder_input=convert2Cuda(decoder_input)
    return decoder_words
def evaluateRandomly(encoder,decoder,n=10):
    for i in range(n):
        pair_idx = random.choice(list(range(len(lang_dataset))))
        pair = lang_dataset.pairs[pair_idx]
        in_lang,out_lang=lang_dataset[pair_idx]
        print('>',pair[0])
        print('=',pair[1])
        output_words = evaluate(encoder,decoder,in_lang)
        output_sentence = ' '.join(output_words)
        print('<',output_sentence)

input_size = lang_dataset.input_lang_words
hidden_size = 256
output_size = lang_dataset.output_lang_words

encoder = Encoder(input_size,hidden_size)
encoder.load_state_dict(t.load('./model/encoder.pth'))
decoder = Decoder(hidden_size,output_size,n_layers=2)
decoder.load_state_dict(t.load('./model/decoder.pth'))

encoder=convert2Cuda(encoder)
decoder=convert2Cuda(decoder)

evaluateRandomly()
