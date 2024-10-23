'''
Document description:
sample sample_num of instances per class
'''

import os
import sys
import csv
import torch
import logging
import argparse
import nltk
from collections import Counter

def parse_args(args):
    parser = argparse.ArgumentParser(description='preprocess data')
    parser.add_argument('--sample_num', type=int, default=5,
                        help='number of classes')
    parser.add_argument("--logging", choices=["INFO", "DEBUG"],
                          default="INFO")
    args = parser.parse_args()
    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(level=args.logging)

    nclass = len(open('classes.txt', 'r').readlines())
    count = [0] * nclass

    f = open('data/train.data', 'r', encoding='utf8')
    fout = open('data/train.data.' + str(args.sample_num), 'w', encoding='utf8')
    for line in f:
        line_sp = line.strip().split('\t')
        if count[int(line_sp[0]) - 1] < args.sample_num:
            fout.write(line)
            count[int(line_sp[0]) - 1] += 1
        if sum(count) == args.sample_num * nclass:
            break
    f.close()
    fout.close()

if __name__ == '__main__':
    main()