from collections import defaultdict
import os
import json
import argparse
import pdb as pdb
from nltk import word_tokenize
import pickle

from tqdm import tqdm

class Voc:
    def __init__(self, name):
        self.name       = name
        self.trimmed    = False
        self.frequented = False
        self.w2id       = {'<s>': 0, '</s>': 1, '<unk>': 2, '<pad>':3}
        self.id2w       = {0: '<s>', 1: '</s>', 2: '<unk>', 3: '<pad>'}
        self.w2c        = defaultdict(int)
        self.vocab_size     = 4

    def addSentence(self, sent):
        sent = self.clean_str(sent)
        sent = word_tokenize(sent)
        for word in sent:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.w2id:
            self.w2id[word]     = self.vocab_size
            self.id2w[self.vocab_size]   = word
            self.w2c[word]      = 1
            self.vocab_size         = self.vocab_size + 1
        else:
            self.w2c[word]      = self.w2c[word] + 1

    def trim(self, mincount):
        if self.trimmed == True:
            return
        self.trimmed    = True

        keep_words = []
        for k, v in self.w2c.items():
            if v >= mincount:
                keep_words += [k]*v

        self.w2id       = {'<s>': 0, '</s>': 1, '<unk>': 2, '<pad>':3}
        self.id2w       = {0: '<s>', 1: '</s>', 2: '<unk>', 3: '<pad>'}
        self.w2c        = defaultdict(int)
        self.vocab_size     = 4
        for word in keep_words:
            self.addWord(word)

    def most_frequent(self, topk):
        if self.frequented == True:
            return
        self.frequented     = True

        keep_words = []
        count      = 3
        sorted_by_value = sorted(self.w2c.items(), key=lambda kv: kv[1], reverse=True)
        for word, freq in sorted_by_value:
            keep_words  += [word]*freq
            count += 1
            if count == topk:
                break

        self.w2id       = {'<s>': 0, '</s>': 1, '<unk>': 2, '<pad>':3}
        self.id2w       = {0: '<s>', 1: '</s>', 2: '<unk>', 3: '<pad>'}
        self.w2c        = defaultdict(int)
        self.vocab_size     = 4
        for word in keep_words:
            self.addWord(word)

    def clean_str(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\?", " ? ", string)

        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()


def main(args):
    filename = 'data/{}/train.json'.format(args.dataset)
    vocab = Voc(args.dataset)

    labels = args.labels.split(",")

    with open(filename, 'r') as f:
        data = json.load(filename)

    for point in tqdm(data['data']):
        for label in labels:
            vocab.add_sentence(point[label])

    vocab.most_frequent(args.vocab_size)

    voc_file = 'data/{}/nltk-vocab.p'.format(args.dataset)

    with open(voc_file, 'wb') as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('File created at {}'.format(voc_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenizer for files')
    parser.add_argument('-dataset', '--dataset', required=True, help='Dataset on which tokenizer needs to be run')
    parser.add_argument('-labels', required=True, type-str, help='Provide comma separated values without spaces')
    parser.add_argument('-vocab_size', '--vocab_size', default=40000, k)
    args = parser.parse_args()

    main(args)
