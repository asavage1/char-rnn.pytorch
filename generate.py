#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import os
import argparse

from helpers import *
from model import *

def generate(decoder, vocab, prime_str=None, predict_len=100, temperature=0.8, cuda=False):  # TODO: change prime_str to be word(s)?
    if prime_str is None:
        prime_str = ['Who', 'is']

    hidden = decoder.init_hidden(300)  # this size is the size of input?, so make it the length of the input (300 for glove word vectors)
    prime_input = []
    for s in prime_str:
        prime_input.append(Variable(char_tensor(s, vocab).unsqueeze(0)))

    if cuda:
        hidden = hidden.cuda()
        prime_input = list(map(lambda x: x.cuda(), prime_input))
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p][0,:], hidden)  # Building up the hidden state one word at a time
        
    inp = prime_input[-1][0,:]  # Takes the previous word to look for the next one
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output[-1].data.view(-1).div(temperature).exp()  # take only the layer of output of the vocabulary
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_word = [vocab[top_i]]  # TODO: words
        predicted += predicted_word
        inp = Variable(char_tensor(predicted_word, vocab).unsqueeze(0))[0,:]
        if cuda:
            inp = inp.cuda()

    return predicted

# Run as standalone script
if __name__ == '__main__':

# Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    decoder = torch.load(args.filename)
    del args.filename
    print(generate(decoder, **vars(args)))

