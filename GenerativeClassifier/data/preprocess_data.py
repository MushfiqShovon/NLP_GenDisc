
import os
import sys
import csv
import torch
import logging
import argparse
import nltk
from collections import Counter

def parse_args(args):
    parser = argparse.ArgumentParser(description='preprocess dbpedia')
    parser.add_argument('--in_datadir', type=str, default='data',
                        help='input data directory, the file name under the \
                        directory should be train.csv, valid.csv, test.csv')
    parser.add_argument('--out_fname', type=str, default='data/data',
                        help='output_prefix')
    parser.add_argument('--num_instance', type=str, default='',
                        help='numer of instance sample from training data')
    parser.add_argument('--vocab_size', type=int, default=None,
                        help='size of vocabulary')
    parser.add_argument('--max_len',  type=int, default=None,
                    help='add <sep> between sentences')
    parser.add_argument("--logging", choices=["INFO", "DEBUG"],
                          default="INFO")
    args = parser.parse_args()
    return args

class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<pad>': 0 , '<unk>': 1, '<sos>': 2, '<eos>': 3} # , '<sep>': 2}
        self.idx2word = ['<pad>', '<unk>', '<sos>', '<eos>'] # , '<sep>']
        self.counter = Counter()

    def add_word(self, word):
        self.counter[word] += 1

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, args):
        self.dictionary = Dictionary()
        self.num_instance = args.num_instance
        if self.num_instance != '':
            self.num_instance = '.' + str(self.num_instance)
        logging.info("tokenize the train file")

        self.build_vocab(args.in_datadir, args.vocab_size, args.max_len)
        self.traindata, self.trainlabel = self.tokenize(os.path.join(args.in_datadir, 'train.data' + self.num_instance), args)
        logging.info("tokenize the valid file")
        self.validdata, self.validlabel = self.tokenize(os.path.join(args.in_datadir, 'valid.data'), args)
        logging.info("tokenize the test file")
        self.testdata, self.testlabel = self.tokenize(os.path.join(args.in_datadir, 'test.data'), args)
        self.vocabsize = self.dictionary.__len__()
        logging.info("number of vocabulary: %d", self.vocabsize)

    def add_vocab(self, filename, max_len):
        f = open(filename, 'r', encoding='utf8')
        for line in f:
            # each row contains class_index, title, and content.
            content = line.strip().split('\t')[1]
            if max_len:
                content = content.split(' ')[:max_len]
            else:
                content = content.split(' ')
            for word in content:
                self.dictionary.add_word(word)

    def build_vocab(self, filedir, vocab_size, max_len):
        self.add_vocab(os.path.join(filedir, 'train.data' + self.num_instance), max_len)
        self.add_vocab(os.path.join(filedir, 'valid.data'), max_len)
        self.add_vocab(os.path.join(filedir, 'test.data'), max_len)

        if vocab_size == None:
            vocab_pair = self.dictionary.counter.most_common()
        else:
            vocab_pair = self.dictionary.counter.most_common(vocab_size - 2)

        for v in vocab_pair:
            self.dictionary.idx2word.append(v[0])
            self.dictionary.word2idx[v[0]] = len(self.dictionary.idx2word) - 1

    def tokenize(self, path, args):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Tokenize file content
        filename = open(path, 'r', encoding='utf8')

        label = []
        sent_ids = []
        max_sent_len = 0

        for line in filename:
            label.append(int(line.strip().split('\t')[0]) - 1)
            content = line.strip().split('\t')[1]
            
            if args.max_len:
                content = ['<sos>'] + content.split()[:args.max_len] + ['<eos>']
            else:
                content = ['<sos>'] + content.split() + ['<eos>']

            if len(content) > max_sent_len:
                max_sent_len = len(content)

            ids = []
            for word in content:
                if word in self.dictionary.idx2word:
                    ids.append(self.dictionary.word2idx[word])
                else:
                    ids.append(self.dictionary.word2idx['<unk>'])
            sent_ids.append(ids)

        logging.info("number of samples: %d", len(label))
        logging.info("max len of samples: %d", max_sent_len)

        return sent_ids, label

def main(args=sys.argv[1:]):
    args = parse_args(args)

    logging.basicConfig(level=args.logging)

    corpus = Corpus(args)

    save_dict = {}
    save_dict['traindata'] = corpus.traindata
    save_dict['trainlabel'] = corpus.trainlabel
    save_dict['validdata'] = corpus.validdata
    save_dict['validlabel'] = corpus.validlabel
    save_dict['testdata'] = corpus.testdata
    save_dict['testlabel'] = corpus.testlabel
    save_dict['vocabsize'] = corpus.vocabsize

    if args.max_len == None:
        args.max_len = 'max'

    args.out_fname = 'data/traindata.v' + str(args.vocab_size) + '.l' + str(args.max_len) + '.s' + str(args.num_instance)

    torch.save(save_dict, args.out_fname)

if __name__ == '__main__':
    main()
