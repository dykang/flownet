import os
import sys
import time
import random
import glob
import codecs
import nltk
from collections import defaultdict
import operator
import numpy as np
from nltk import Tree
from nltk.tokenize import sent_tokenize
from utils import INFO_LOG, flatten_tree


def paragraph_tokenizer(line):
    return [[w for w in sent.split(' ') if w] for sent in line]


class Vocab(object):
    def __init__(
            self,
            is_paragraph=False,
            is_rst=False,
            max_vocab=False,
            is_lower=False):
        self.is_paragraph = is_paragraph
        self.is_rst = is_rst
        self.max_vocab = max_vocab
        self.is_lower = is_lower

        self.PAD = "<pad>"
        self.UNK = "<unk>"
        #self.BOS = "<s>"
        self.EOS = "</s>"

    def buildFromFiles(self, files):
        if not isinstance(files, list):
            raise ValueError("buildFromFiles input type error")

        INFO_LOG("build vocabulary from files ...")
        # {self.BOS: 0, self.EOS: 0, self.UNK: 0}
        self.word_cnt = defaultdict(lambda: 0)
        # {self.BOS: 0, self.EOS: 0, self.UNK: 0}
        self.rst_cnt = defaultdict(lambda: 0)
        for _file in files:
            line_num = 0
            for line in open(_file):
                tks = line.strip().split('\t')
                line_num += 1

                if self.is_paragraph:
                    sents = tks[4:]
                    if self.is_lower:
                        sents = [sent.strip().lower() for sent in sents]
                    sents = paragraph_tokenizer(sents)
                    for sent in sents:
                        for w in sent:
                            self.word_cnt[w] += 1

                elif self.is_rst:
                    sents = tks[4:-1]
                    rst = tks[-1].strip()

                    # count words
                    if self.is_lower:
                        sents = [sent.strip().lower() for sent in sents]
                    sents = paragraph_tokenizer(sents)
                    for sent in sents:
                        for w in sent:
                            self.word_cnt[w] += 1

                    # count RST relations
                    try:
                        t = Tree.fromstring(rst)
                        rel = flatten_tree(t)

                    except Exception as e:
                        continue

                    for phrase in rel:
                        if self.is_RST(phrase):
                            self.rst_cnt[phrase] += 1

                else:
                    print('Wrong type', self.is_paragraph, self.is_rst)
                    sys.exit(1)

        count_pairs = sorted(self.word_cnt.items(),
                             key=lambda x: (-x[1], x[0]))
        self.words, _ = list(zip(*count_pairs))

        #self.words = [self.PAD, self.BOS,  self.UNK] + list(self.words)
        self.words = [self.PAD, self.UNK, self.EOS] + list(self.words)

        if self.max_vocab:
            print('Vocab size :%d -> %d' % (len(self.words), self.max_vocab))
            self.words = self.words[:self.max_vocab]
        INFO_LOG("vocab size: {}".format(self.size()))

        if self.is_rst:
            count_rst_pairs = sorted(
                self.rst_cnt.items(), key=lambda x: (-x[1], x[0]))
            self.rsts, _ = list(zip(*count_rst_pairs))
            self.rsts = [self.PAD, self.UNK] + list(self.rsts)
            INFO_LOG("RST size: {}".format(len(self.rsts)))
            self.rst2id = dict(zip(self.rsts, range(len(self.rsts))))
            INFO_LOG("RSTs {}".format(
                ' '.join(['%d:%s' % (i, self.rsts[i]) for i in range(10)])))

        self.word2id = dict(zip(self.words, range(len(self.words))))
        self.UNK_ID = self.word2id[self.UNK]
        INFO_LOG("vocabs {}".format(
            ' '.join(['%d:%s' % (i, self.words[i]) for i in range(10)])))

    def is_RST(self, phrase):
        tks = phrase.strip().split(' ')
        if len(tks) == 1 and (
                phrase.startswith('NS') or
                phrase.startswith('SN') or
                phrase.startswith('NN')):
            return True
        else:
            return False

    def encode(self, line):
        tks = line.strip().split('\t')

        if self.is_paragraph:

            sents = tks[4:]
            if self.is_lower:
                sents = [sent.strip().lower() for sent in sents]
            sents = paragraph_tokenizer(sents)
            new_sents = []
            for sent in sents:
                sent_ids = [
                    self.word2id[w] if w in self.word2id else self.UNK_ID for w in sent]
                #sent_ids = [self.word2id[self.BOS]] + sent_ids + [self.word2id[self.EOS]]
                sent_ids = sent_ids
                new_sents.append(sent_ids)
            return new_sents

        elif self.is_rst:
            sents = tks[4:-1]
            rst = tks[-1].strip()
            plen = int(tks[2])

            # encode words
            if self.is_lower:
                sents = [sent.strip().lower() for sent in sents]
            sents = paragraph_tokenizer(sents)
            original_sents = []
            for sent in sents:
                sent_ids = [
                    self.word2id[w] if w in self.word2id else self.UNK_ID for w in sent]
                original_sents.append(sent_ids)

            # encode RST
            try:
                t = Tree.fromstring(rst)
                rst = flatten_tree(t)
            except Exception as e:
                # print e
                return None, None, None

            # get word+rst sequence [[w1,w2..],rst1,[w1,w2...],rst2, ....]
            rst_seqs = []
            for phrase in rst:
                if self.is_RST(phrase):
                    rst_seqs.append(
                        self.rst2id[phrase] if phrase in self.rst2id else self.rst2id[self.UNK])
                else:
                    phrase = phrase.strip()
                    if self.is_lower:
                        phrase = phrase.lower()
                    tks = phrase.split(' ')
                    sent_ids = [
                        self.word2id[w] if w in self.word2id else self.UNK_ID for w in tks]
                    rst_seqs.append(sent_ids)

            # delete the consecutive RSTs or consecutive sents
            sents, rsts = [], []
            for sid in range(len(rst_seqs)):
                if sid == 0:
                    sents.append(rst_seqs[sid])
                else:
                    if type(rst_seqs[sid]) == type(rst_seqs[sid - 1]) == int:
                        continue
                    if isinstance(rst_seqs[sid], list):
                        sents.append(rst_seqs[sid])
                    else:
                        if sid == len(rst_seqs) - 1:
                            continue
                        rsts.append(rst_seqs[sid])

            if len(rsts) + 1 != len(sents):
                # print '[WRONG-RST=SENTS]',len(rsts),len(sents)
                return None, None, None

            if np.sum([len(s) for s in sents]) != np.sum([len(s)
                                                          for s in original_sents]):
                return None, None, None

            # flatten rsts and sents
            sents_flatten, rsts_flatten = [], []
            for pid, sent in enumerate(sents):
                sents_flatten += sent
                rst_rel = [self.rst2id[self.PAD]] * len(sent)
                if pid > 0:
                    rst_rel[0] = rsts[pid - 1]
                rsts_flatten += rst_rel

            assert len(sents_flatten) == len(rsts_flatten)

            # map RST parses into original_sents (i.e same paragraph number)
            split_points = [len(s) for s in original_sents]
            split_points = np.cumsum(split_points) - 1
            sents_final, rsts_final = [], []
            sents_one, rsts_one = [], []
            for id, (w, r) in enumerate(zip(sents_flatten, rsts_flatten)):
                if id in split_points:

                    #sent_ids = [self.word2id[self.BOS]] + sent_ids + [self.word2id[self.EOS]]
                    sents_final.append(sents_one)
                    rsts_final.append(rsts_one)
                    sents_one, rsts_one = [], []
                else:
                    sents_one.append(w)
                    rsts_one.append(r)

            return plen, sents_final, rsts_final
        else:
            sentence = line.strip().split()
            return [self.word2id[w]
                    if w in self.word2id else self.UNK_ID for w in sentence]

    def decode(self, ids):
        return [self.words[_id] for _id in ids]

    def size(self):
        return len(self.words)



class ReaderFT(object):
    def __init__(self, data_path, dataset, max_vocab=False, verbose=False):
        self.verbose = verbose
        self.train_file = os.path.join(
            data_path,
            '%s_delta.txt.train' %
            (dataset))
        self.valid_file = os.path.join(
            data_path,
            '%s_delta.txt.valid' %
            (dataset))
        self.test_file = os.path.join(
            data_path,
            '%s_delta.txt.test' %
            (dataset))

        self.vocab = Vocab(
            is_paragraph=True,
            max_vocab=max_vocab,
            is_lower=True)
        self.vocab.buildFromFiles([self.train_file])

    def getVocabSize(self):
        return self.vocab.size()

    def yieldSpliceBatch(
            self,
            tag,
            batch_size,
            step_size,
            para_len=False,
            shuffle=False,
            limit=False):
        eos_index = self.vocab.word2id[self.vocab.EOS]
        unk_index = self.vocab.word2id[self.vocab.UNK]
        pad_index = self.vocab.word2id[self.vocab.PAD]

        if tag == 'train':
            _file = self.train_file
        elif tag == 'valid':
            _file = self.valid_file
        elif tag == 'test':
            _file = self.test_file
        else:
            print('Error', tag)
            sys.exit(1)

        if self.verbose:
            INFO_LOG("File: %s" % _file)

        data = []
        total_token = 0
        for line in open(_file):
            sents = self.vocab.encode(line)
            data.append(sents)
            total_token += np.sum([len(s) for s in sents])
            #data += self.vocab.encode(line) + [eos_index]

        if shuffle:
            random.shuffle(data)
        paragraph_num = len(data)

        batch_len = paragraph_num // batch_size
        batch_num = (batch_len - 1)  # // step_size
        if batch_num == 0:
            raise ValueError(
                "batch_num == 0, decrease batch_size or step_size")

        if limit:
            batch_num = limit

        if self.verbose:
            INFO_LOG(
                "  {} pragraphs, {} tokens".format(
                    paragraph_num, total_token))
            used_token = batch_num * batch_size  # * step_size
            INFO_LOG("  {} batches, {}*{} = {}({:.2%}) tokens will be used".format(batch_num,
                                                                                   batch_num, batch_size, used_token, float(used_token) / paragraph_num))

        # first: [batch x step], last: [batch x step],
        # middle: [batch x plen x step]
        for batch_id in range(batch_num):
            index = batch_id * batch_size
            batch = data[index: index + batch_size]

            batch_padded = []
            word_cnt = 0
            for para in batch:
                para_padded = []
                for pid, sent in enumerate(para):
                    padding = [pad_index] * (step_size - len(sent))
                    sent = sent + padding
                    if pid > 0 and pid < len(para) - 1:
                        symbol_cnt = np.sum(
                            [1 for wid in sent[:step_size] if wid < 3])
                        word_cnt += step_size - symbol_cnt  # len(padding)
                    para_padded.append(sent[:step_size])
                batch_padded.append(para_padded)
            batch = np.array(batch_padded)
            middle = batch[:, 1:-1, :]
            target = np.reshape(middle, (batch_size, -1))[:, 1:]
            target = np.reshape(
                np.array([list(t) + [0] for t in target]), (batch_size, -1, step_size))

            batch = {
                'xfirst': batch[:, 0, :],
                'xlast': batch[:, -1, :],
                'xmiddles': middle,
                'targets': target,
            }
            yield batch_id, batch_num, word_cnt, batch


class ReaderRST(object):
    def __init__(self, data_path, dataset, max_vocab=False, verbose=False):
        self.verbose = verbose
        self.train_file = os.path.join(data_path, '%s.train' % (dataset))
        self.valid_file = os.path.join(data_path, '%s.valid' % (dataset))
        self.test_file = os.path.join(data_path, '%s.test' % (dataset))

        self.vocab = Vocab(is_rst=True, max_vocab=max_vocab, is_lower=True)
        self.vocab.buildFromFiles([self.train_file, self.valid_file])

    def getVocabSize(self):
        return self.vocab.size()

    def yieldSpliceBatch(
            self,
            tag,
            batch_size,
            step_size,
            para_len=False,
            model_name=False,
            shuffle=False,
            limit=False):

        if tag == 'train':
            _file = self.train_file
        elif tag == 'valid':
            _file = self.valid_file
        elif tag == 'test':
            _file = self.test_file
        else:
            print('Error', tag)
            sys.exit(1)

        if self.verbose:
            INFO_LOG("File: %s" % _file)

        data = []
        cnt_wrong = 0
        total_token, total_rst = 0, 0
        for line in open(_file):
            plen, sents, rsts = self.vocab.encode(line)
            if sents is None or rsts is None:
                cnt_wrong += 1
                continue

            data.append((plen, sents, rsts))
            total_token += np.sum([len(s) for s in sents])
            total_rst += np.sum([np.count_nonzero(r) for r in rsts])

        if shuffle:
            random.shuffle(data)

        if para_len == 'all':
            data.sort(key=operator.itemgetter(0))

        paragraph_num = len(data)
        batch_len = int(paragraph_num / batch_size)
        batch_num = batch_len  # (batch_len - 1) #// step_size
        if batch_num == 0:
            raise ValueError(
                "batch_num == 0, decrease batch_size or step_size")

        if limit:
            batch_num = limit

        if self.verbose:
            INFO_LOG("  {} pragraphs, {} tokens {} rsts {} wrong cases".format(
                paragraph_num, total_token, total_rst, cnt_wrong))
            used_token = batch_num * batch_size  # * step_size
            INFO_LOG("  {} batches, {}*{} = {}({:.2%}) tokens will be used".format(batch_num,
                                                                                   batch_num, batch_size, used_token, float(used_token) / paragraph_num))

        # first: [batch x step], last: [batch x step],
        # middle: [batch x plen x step]
        for batch_id in range(batch_num):
            index = batch_id * batch_size
            batch = data[index: index + batch_size]

            # check whether all lengths are same for the batch
            plens = [b[0] for b in batch]
            if not all(plens[0] == l for l in plens):
                print('Not same length', plens)
                import pdb
                pdb.set_trace()
                sys.exit(1)
            plen = plens[0]

            sents_batch = []
            rsts_batch = []
            lens_batch = []
            plen_batch = []
            word_cnt = 0
            for _, sents, rsts in batch:
                sents_padded, rsts_padded, lens = [], [], []
                # start padding
                for pid, (sent, rst) in enumerate(zip(sents, rsts)):
                    lens.append(len(sent[:step_size]))

                    sent_padding = [
                        self.vocab.word2id[self.vocab.PAD]] * (step_size - len(sent))
                    sent = sent + [2] + sent_padding
                    sent = sent[:step_size]
                    sents_padded.append(sent)

                    rst_padding = [
                        self.vocab.rst2id[self.vocab.PAD]] * (step_size - len(rst))
                    rst = rst + [0] + rst_padding
                    rst = rst[:step_size]
                    rsts_padded.append(rst)

                    # count sequence length and word_count for perplexity
                    if model_name in ['rnn']:
                        word_cnt += np.sum([1 for wid in sent if wid >= 3])
                    elif model_name in ['hred']:
                        if pid > 0:
                            word_cnt += np.sum([1 for wid in sent if wid >= 3])
                    elif model_name in ['textplan', 'ftrnn']:
                        # don't count first and last
                        if pid > 0 and pid < len(sents) - 1:
                            word_cnt += np.sum([1 for wid in sent if wid >= 3])

                sents_batch.append(sents_padded)
                rsts_batch.append(rsts_padded)
                lens_batch.append(lens)

            sents_batch = np.array(sents_batch)
            rsts_batch = np.array(rsts_batch)
            lens_batch = np.array(lens_batch)

            batch = {
                'sents': sents_batch,
                'rsts': rsts_batch,
                'seq_lens': lens_batch,
                'para_len': plen
            }
            yield batch_id, batch_num, word_cnt, batch

