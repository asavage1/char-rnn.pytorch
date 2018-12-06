# https://github.com/spro/char-rnn.pytorch

import unidecode
import string
import random
import time
import math
import re
import torch
import spacy
import numpy as np

nlp = spacy.load('en_core_web_md')
word2vec = lambda word: nlp.vocab[word].vector

# Reading and un-unicode-encoding data

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    file = re.findall(r"[\w']+", file)
    return file, len(file)

def read_file2(filename):
    qa_pairs = []
    with open(filename, 'r') as f:
        qa_pairs = [line.rstrip() for line in f.readlines()]

    qa_pairs = list(zip(qa_pairs, qa_pairs[1:]))
    qa_pairs = [qa_pairs[i] for i in range(0, len(qa_pairs), 2)]

    return qa_pairs


# Turning a string into a tensor

def char_tensor(string, all_words):
    string_vec = list(map(word2vec, string))
    tensor = np.sum(string_vec, axis=0)
    tensor = np.vectorize(lambda x: round(x - np.min(tensor)))(tensor)  # Normalize to be on scale [0, vocab_size)
    return torch.tensor(tensor).long()

    # tensor = torch.zeros(len(string)).long()
    # for c in range(len(string)):
    #     try:
    #         tensor[c] = all_words.index(string[c])
    #     except:
    #         continue
    # return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

