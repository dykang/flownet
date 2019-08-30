# -*- coding: utf-8 -*-
import os
import sys
import glob
import re
import nltk
import pprint
import codecs
import argparse
import codecs
import io
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.moses import MosesTokenizer, MosesDetokenizer
import numpy as np
pp = pprint.PrettyPrinter(indent=4)

reload(sys)
sys.setdefaultencoding('utf-8')

mtknr = MosesTokenizer(lang='en')
detokenizer = MosesDetokenizer()


def detokenize(sent):
    return detokenizer.detokenize(sent, return_str=True)


def basic_tokenizer(sent):
    return sent.split()


def better_tokenizer(sent):
    try:
        # aggressive_dash_splits=True,
        return mtknr.tokenize(sent, return_str=True).split()
        # return word_tokenize(sent)
    except Exception as e:
        # print sent,e
        return None


def sentence_tokenizer(sents,
                       min_sent=4, max_sent=8,
                       min_sent_len=False, max_sent_len=False):
    #sents = sent_tokenize(unicode(sents, errors='replace'))
    sents = sent_tokenize(sents)
    sents = [better_tokenizer(sent) for sent in sents]
    for sent in sents:
        if sent is None:
            return None
    if len(sents) < min_sent:
        return None
    if len(sents) > max_sent:
        return None

    if min_sent_len:
        for sent in sents:
            if len(sent) < min_sent_len:
                return None
    if max_sent_len:
        for sent in sents:
            if len(sent) > max_sent_len:
                return None

    return [word_tokenize(' '.join(sent)) for sent in sents]


def main(args, limit=False):

    fout_dic = {}
    for len_sent in range(args.min_sent, args.max_sent + 1):
        fout_dic[len_sent] = open('%s_%d.txt' % (args.out_file, len_sent), 'w')

    cnt = 0
    topic = ''
    topic_changed = False
    total_file = len(
        sorted(
            glob.glob(
                os.path.join(
                    args.data_dir) +
                'Topics/*/*.txt')))
    for fid, file in enumerate(
        sorted(
            glob.glob(
            os.path.join(
                args.data_dir) + 'Topics/*/*.txt'))):
        with codecs.open(file, 'r', encoding='utf-8', errors='replace') as fin:
            tks = file.split('/')
            if tks[-2] != topic:
                topic_changed = True
            else:
                topic_changed = False

            topic = tks[-2]
            file_prefix = tks[-1].split('.')[0]
            import pdb
            pdb.set_trace()
            if topic_changed:
                print '[%d/%d] Reading topic.. %s' % (fid, total_file, topic)

            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                sents = sentence_tokenizer(
                    line,
                    min_sent=args.min_sent,
                    max_sent=args.max_sent
                )
                if sents is None:
                    continue
                cnt += 1
                avg_len = np.average([len(sent) for sent in sents])

                # topic \t file_prefix \t sent-len \t sent1 \t sent2 \t sent3
                # ... \n
                fout = fout_dic[len(sents)]
                fout.write('%s\t%s\t%d\t%.2f\t%s\n' % (topic, file_prefix, len(
                    sents), avg_len, '\t'.join([' '.join(sent) for sent in sents])))
                if cnt % 10000 == 0:
                    print '\t%d paragraphs, %d files ..' % (cnt, fid)
                    fout.flush()
                if limit and cnt == limit:
                    sys.exit(1)

    for len_sent, fout in fout_dic.items():
        fout.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='This is a PyMOTW sample program')
    parser.add_argument('-data_dir')
    parser.add_argument('-out_file')
    parser.add_argument('-max_sent', default=10, type=int)
    parser.add_argument('-min_sent', default=2, type=int)
    args = parser.parse_args()
    pprint.pprint(vars(args), width=1)
    main(args)
