# -*- coding: utf-8 -*-
import re

def filter_sent(sent):
    #import pdb; pdb.set_trace()
    # print sent
    sent = sent.replace('& quot ;', '"')
    sent = sent.replace('& apos ;', '\'')
    sent = sent.replace('& amp ;', '')
    sent = sent.replace('& #124 ;', '')
    sent = sent.replace('& lt ;', '')
    sent = sent.replace('& gt ;', '')
    sent = sent.replace('& #91 ;', '')
    sent = sent.replace('& #93 ;', '')
    sent = sent.replace('”', '"')
    sent = sent.replace('“', '"')
    # print '>>',sent
    re.sub(' +', ' ', sent)
    return sent


data_dir = '/data/paragen/data/old/v2/'


for dtype in ['nyt', 'book', 'arxiv']:
    fout = open(data_dir + '../../%s.txt' % (dtype), 'w')
    for l in range(4, 8, 1):
        cnt = 0
        line_cnt = 0
        for line in open(data_dir + '%s_%d_delta.txt' % (dtype, l), 'r'):
            line_cnt += 1
            tks = line.strip().split('\t')
            cat1, cat2, para_len, sent_len = tks[:4]
            sents = tks[4:]
            para = [[w for w in sent.split(' ')] for sent in sents]

            # filter "& apos ;"
            para = [filter_sent(' '.join(sent)).split(' ') for sent in para]

            # filter non period ending sentence
            if not para[-1][-1] in ['.', '?', '!', '"', '\'']:
                cnt += 1
            else:
                fout.write('%s\t%s\n' % (
                    '\t'.join(tks[:4]),
                    '\t'.join([' '.join(sent) for sent in para])))
        print l, dtype, cnt, line_cnt
    fout.close()

    #import pdb; pdb.set_trace()
