#coding=utf-8
"""
@version=1.0
@author:zsh
@file:main.py
@time:2019/1/8 17:30
"""

import torch as t
import random,time,pickle,os
from torch import nn,optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from util import tensorFromPair,prepareData,timeSince,showPlot,TextDataset
from model import Encoder,Decoder
MAX_LENGTH=10

SOS_token = 0
EOS_token = 1


def convert2Cuda(data):
    return data.cuda() if t.cuda.is_available() else data

def trainData():
    device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
    lang_dataset = TextDataset()
    lang_dataloader = DataLoader(lang_dataset, shuffle=True)
    input_size = lang_dataset.input_lang_words
    hidden_size = 256
    output_size = lang_dataset.output_lang_words
    total_epoch = 20
    use_attn = False
    # n_iters=10000
    # training_pair = [tensorFromPair(input_lang,output_lang,random.choice(pairs)) for i in range(n_iters)]
    encoder = convert2Cuda(Encoder(input_size,hidden_size))
    decoder = convert2Cuda(Decoder(hidden_size,output_size,n_layers=2))

    param = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(param, 1e-3)
    plot_losses = []
    criterion = nn.NLLLoss()

    for epoch in range(total_epoch):
        start = time.time()
        running_loss = 0
        print_loss_total = 0
        total_loss = 0

        for i, data in enumerate(lang_dataloader):
            in_lang, out_lang = data
            in_lang = convert2Cuda(in_lang)
            out_lang = convert2Cuda(out_lang)
            in_lang = Variable(in_lang)
            out_lang = Variable(out_lang)
            encoder_outputs = Variable(
                t.zeros(MAX_LENGTH, encoder.hidden_size)
            )
            encoder_outputs = convert2Cuda(encoder_outputs)
            encoder_hidden = encoder.initHidden()
            for ei in range(in_lang.size(1)):
                encoder_output, encoder_hidden = encoder(in_lang[:, ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0][0]
            decoder_input = Variable(t.LongTensor([[SOS_token]]))
            decoder_input = convert2Cuda(decoder_input)
            decoder_hidden = encoder_hidden

            loss = 0

            if not use_attn:
                for di in range(out_lang.size(1)):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    loss += criterion(decoder_output, out_lang[:, di])
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi[0][0]
                    decoder_input = Variable(t.LongTensor([[ni]]))
                    decoder_input = convert2Cuda(decoder_input)
                    if ni == EOS_token:
                        break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            print_loss_total += loss.data[0]
            total_loss += loss.data[0]

            if (i + 1) % 100 == 0:
                print('{}/{},Loss:{:.6f}'.format(i + 1, len(lang_dataloader), running_loss / 10))
                running_loss=0
            if (i + 1) % 10 == 0:
                plot_loss = print_loss_total / 100
                plot_losses.append(plot_loss)
                print_loss_total = 0
        during = time.time() - start
        print('Finish {}/{},Loss:{:.6f}.,Time:{:.0f}s\n'.format(epoch + 1, total_epoch, total_loss / len(lang_dataset),during))
    showPlot(plot_losses)
    t.save(encoder.state_dict(),'./model/encoder.pth')
    t.save(decoder.state_dict(),'./model/decoder.pth')
if __name__ == '__main__':


    trainData()