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
import spacy
from collections import Counter
from models.Paper import Paper
from models.Review import Review
from models.ScienceParse import ScienceParse
from models.ScienceParseReader import ScienceParseReader
from shutil import copyfile, rmtree
pp = pprint.PrettyPrinter(indent=4)

from prepare_book import sentence_tokenizer, better_tokenizer, basic_tokenizer

def main(args, limit=False):

    fout_dic = {}
    for len_sent in range(args.min_sent, args.max_sent + 1):
        fout_dic[len_sent] = open('%s_%d.txt' % (args.out_file, len_sent), 'w')

    review_files = glob.glob(args.data_dir + '/reviews' + '/*.json')
    print 'Number of papers:', len(review_files)

    cnt = 0
    topic = ''
    topic_changed = False
    category_types = ['cs.cl', 'cs.lg', 'cs.ai']

    #nlp = spacy.load('en', parser=False)
    for rid, review_file in enumerate(review_files):

        if rid % 1000 == 0:
            print '[%d/%d]' % (rid, len(review_files))

        paper = Paper.from_json(review_file)
        if not paper:
            continue

        paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(
            paper.ID, paper.TITLE, paper.ABSTRACT, args.data_dir + '/scienceparse/')

        # paper ID
        file_prefix = paper.ID

        # sentences
        sections = paper.SCIENCEPARSE.get_sections_dict()

        for topic, content in sections.items():

            paragraphs = content.split('\n')

            for paragraph in paragraphs:
                if paragraph == '':
                    continue
                sents = sentence_tokenizer(
                    paragraph,
                    min_sent=args.min_sent,
                    max_sent=args.max_sent,
                    min_sent_len=args.min_sent_len,
                    max_sent_len=args.max_sent_len
                )
                if sents is None:
                    continue
                cnt += 1
                avg_len = np.average([len(sent) for sent in sents])

                fout = fout_dic[len(sents)]
                fout.write('%s\t%s\t%d\t%.2f\t%s\n' % (topic, file_prefix, len(
                    sents), avg_len, '\t'.join([' '.join(sent) for sent in sents])))
                if cnt % 1000 == 0:
                    print '\t%d paragraphs, %d files ..' % (cnt, rid)
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
    parser.add_argument('-max_sent', default=7, type=int)
    parser.add_argument('-min_sent', default=4, type=int)
    parser.add_argument('-max_sent_len', default=40, type=int)
    parser.add_argument('-min_sent_len', default=5, type=int)

    args = parser.parse_args()
    pprint.pprint(vars(args), width=1)
    main(args)
