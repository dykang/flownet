import gensim
import time
import os
import sys
import random
import collections
import functools
import pprint
import nltk
import re
import pickle as pkl
import numpy as np
import tensorflow as tf
pp = pprint.PrettyPrinter()


def LOAD_READER(flags, config):
    from reader import ReaderFT, ReaderRST
    dataset_name, para_len = flags.dataset.split('_')
    setattr(config, 'para_len', int(para_len))
    reader = ReaderRST
    return reader(
        flags.data_path,
        flags.dataset,
        max_vocab=config.max_vocab,
        verbose=True)


def LOAD_CONFIG_MODEL(flags):  # model_name):
    from config import Seq2SeqConfig, RSTSeq2SeqConfig
    from config import RNNConfig, HRNNConfig, HREDConfig, DHRNNConfig
    from models.lm_models import RNN, HRNN, HRED, DHRNN
    from models.clm_models import Seq2Seq, RSTSeq2Seq

    if flags.model_name == 'seq2seq':
        config, model = Seq2SeqConfig, Seq2Seq
    elif flags.model_name == 'rstseq2seq':
        config, model = RSTSeq2SeqConfig, RSTSeq2Seq
    elif flags.model_name == 'hrnn':
        config, model = HRNNConfig, HRNN
    elif flags.model_name == 'rnn':
        config, model = RNNConfig, RNN
    elif flags.model_name == 'hred':
        config, model = HREDConfig, HRED
    elif flags.model_name == 'dhrnn':
        config, model = DHRNNConfig, DHRNN
    else:
        print('Wrong', flags.model_name)
        sys.exit(1)

    setattr(config, 'dataset', flags.dataset)
    setattr(config, 'reload', flags.reload)
    setattr(config, 'is_test', flags.is_test)
    return config, model


def LOAD_TF_CONFIG():
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    return tf_config


def INFO_LOG(info):
    print("[%s] %s" % (time.strftime("%Y%m%d %X", time.localtime()), info))


def load_embedding(config, embedding_path, vocab, verbose=True, limit=None):
    if config.use_embeddings and embedding_path:
        if verbose:
            print("loading embeddings from {}".format(embedding_path))
        vocab_dict = import_embeddings(embedding_path, limit=limit)
        config.embedding_size = len(vocab_dict["the"])
        embedding_var = np.random.normal(
            0.0, config.init_scale, [
                config.vocab_size, config.embedding_size])
        no_embeddings = 0
        for word in vocab:
            try:
                embedding_var[vocab[word], :] = vocab_dict[word]
            except KeyError:
                no_embeddings += 1
                continue
        if verbose:
            print("num embeddings with no value:{}".format(no_embeddings))
        del vocab_dict
        return embedding_var
    else:
        return None


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def progress(progress, status=""):
    barLength = 10
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Finished.\r\n"

    block = int(round(barLength * progress))
    text = "\rPercent: [%s] %.2f%% | %s" % (
        "#" * block + " " * (barLength - block), progress * 100, status)

    sys.stdout.write(text)
    sys.stdout.flush()


def write_f(output, verbose=False, fwrite=False):
    if verbose:
        print(output)
    if fwrite:
        fwrite.write('%s\n' % (output))
        fwrite.flush()


def import_embeddings(filename, limit=None):
    glove_embedding = gensim.models.KeyedVectors.load_word2vec_format(
        filename, binary=True, limit=limit)
    return glove_embedding


class LearningRateUpdater(object):
    def __init__(self, init_lr, decay_rate, decay_when):
        self._init_lr = init_lr
        self._decay_rate = decay_rate
        self._decay_when = decay_when
        self._current_lr = init_lr
        self._last_ppl = -1

    def get_lr(self):
        return self._current_lr

    def update(self, cur_ppl):
        if self._last_ppl > 0 and self._last_ppl - cur_ppl < self._decay_when:
            current_lr = self._current_lr * self._decay_rate
            INFO_LOG(
                "learning rate: {} ==> {}".format(
                    self._current_lr,
                    current_lr))
            self._current_lr = current_lr
        self._last_ppl = cur_ppl


def flatten_tree(tree):
    if tree.label() == 'EDU':
        leaves = []
        for st in tree:
            if isinstance(st, nltk.tree.Tree):
                new_st = re.split(r'(\W)', str(st))
                new_st = [w for w in new_st if w.strip()]
                leaves += new_st
            else:
                leaves += [st]
        return [' '.join(leaves)]

    else:
        is_leaf = []
        cnt = 0
        for subtree in tree:
            if isinstance(subtree, nltk.tree.Tree):
                is_leaf += flatten_tree(subtree)
                if cnt == 0:
                    is_leaf += [tree.label()]
                cnt += 1
        return is_leaf  # + [tree.label()]
