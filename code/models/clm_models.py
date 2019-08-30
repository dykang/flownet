import os
import sys
import time
import numpy as np
import tensorflow as tf
from .base import Base


class Seq2Seq(Base):
    def __init__(
            self,
            config,
            mode,
            pretrained_embeddings=None,
            device='/gpu:0',
            reuse=None):
        self.config = config
        self.device = device
        self.mode = mode
        self.reuse = reuse
        self.pretrained_embeddings = pretrained_embeddings
        self.name = 'Seq2Seq'

        if mode == "train":
            self.is_training = True
            self.batch_size = self.config.train_batch_size
            self.step_size = self.config.train_step_size
            # * (self.config.para_len - 2)
            self.step_infer_size = self.step_size
        elif mode == "valid":
            self.is_training = False
            self.batch_size = self.config.valid_batch_size
            self.step_size = self.config.valid_step_size
            # * (self.config.para_len - 2)
            self.step_infer_size = self.step_size
        elif mode == 'test':
            self.is_training = False
            self.batch_size = self.config.test_batch_size
            self.step_size = self.config.test_step_size
            # * (self.config.para_len - 2)
            self.step_infer_size = self.step_size
        else:
            print('wrong', mode)
            sys.exit(1)

        self.dtype = tf.float32
        self.vocab_size = self.config.vocab_size
        self.embed_dim = self.config.word_embedding_dim
        self.lstm_size = self.config.lstm_size
        self.lstm_layers = self.config.lstm_layers
        self.update_embeddings = self.config.update_embeddings

        self.para_len = self.config.para_len
        self.dataset = self.config.dataset
        self.reload = self.config.reload
        self.checkpoint_dir = "/data/tf/ckpts/"
        self.log_dir = "/data/tf/logs/"

        self._attrs = [
            'batch_size',
            'para_len',
            'vocab_size',
            'step_size',
            'embed_dim',
            'lstm_size',
            'lstm_layers',
            'update_embeddings']

        std_vocab = np.sqrt(1. / self.vocab_size)
        self.initializer_emb = tf.random_uniform_initializer(
            -std_vocab, std_vocab)
        self.initializer_vocab = tf.random_uniform_initializer(
            -std_vocab * 0.8, std_vocab * 0.8)
        self.initializer_constant = tf.constant_initializer(
            0.0, dtype=self.dtype)

        self.W_lm = tf.get_variable(
            "W_lm", [
                self.lstm_size * 2, self.vocab_size], initializer=self.initializer_vocab)
        self.b_lm = tf.get_variable("b_lm",
                                    [self.vocab_size],
                                    initializer=self.initializer_constant)

        # RNN Cell
        cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
        if self.is_training and config.dropout_prob > 0:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=1. - config.dropout_prob)
        self.cell_enc = tf.nn.rnn_cell.MultiRNNCell(
            [cell] * self.lstm_layers, state_is_tuple=True)

        cell = tf.contrib.rnn.LSTMCell(self.lstm_size * 2, state_is_tuple=True)
        if self.is_training and config.dropout_prob > 0:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=1. - config.dropout_prob)
        self.cell_dec = tf.nn.rnn_cell.MultiRNNCell(
            [cell] * self.lstm_layers, state_is_tuple=True)

    def init(self):
        print('Initializing...')
        with tf.device(self.device), tf.name_scope(self.mode), tf.variable_scope("Seq2Seq", reuse=self.reuse):
            self.placeholders()
            enc_state = self.encoder()
            dec_states, dec_outputs, states = self.decoder(enc_state)
            self.loss_fn(dec_outputs)

            self.global_step = tf.Variable(
                0, name='global_step', trainable=False)
            if self.is_training:
                self.lr = tf.Variable(0.0, trainable=False)
                self.train_op = self.add_train_op(
                    'adagrad',
                    self.lr,
                    self.loss,
                    clip=self.config.max_grad_norm)
            else:
                self.train_op = tf.no_op()

    def placeholders(self):
        # Placeholders
        self.xfirst_sents = tf.placeholder(
            tf.int32, [self.batch_size, self.step_size])
        self.xfirst_lens = tf.placeholder(tf.int32, [self.batch_size])
        self.xlast_sents = tf.placeholder(
            tf.int32, [self.batch_size, self.step_size])
        self.xlast_lens = tf.placeholder(tf.int32, [self.batch_size])

        self.xmiddles_sents, self.xmiddles_lens = [], []
        self.targets_sents, self.targets_lens = [], []

        for i in range(self.para_len - 2):
            self.xmiddles_sents.append(
                tf.placeholder(
                    tf.int32, [
                        self.batch_size, self.step_infer_size], name='xmiddles_sents%d' %
                    (i)))
            self.xmiddles_lens.append(
                tf.placeholder(
                    tf.int32, [
                        self.batch_size], name='xmiddles_lens%d' %
                    (i)))
            self.targets_sents.append(
                tf.placeholder(
                    tf.int32, [
                        self.batch_size, self.step_infer_size], name='targets_sents%d' %
                    (i)))
            self.targets_lens.append(
                tf.placeholder(
                    tf.int32, [
                        self.batch_size], name='targets_lens%d' %
                    (i)))
        self.initial_state = tf.placeholder(
            tf.float32, [self.lstm_layers, 2, self.batch_size, self.lstm_size])

    def encoder(self):
        # WORD EMBEDDING [batch x step] x [Vocab x emb_size]
        with tf.variable_scope('embedding'):
            # Word embedding
            if self.pretrained_embeddings is not None:
                self.word_embedding = tf.get_variable(
                    'word_embedding', [self.vocab_size, self.embed_dim],
                    trainable=self.config.update_embeddings,
                    initializer=self.initializer_emb)
                self.embedding_placeholder = tf.placeholder(
                    tf.float32, [self.vocab_size, self.embed_dim])
                self.embedding_init = self.word_embedding.assign(
                    self.embedding_placeholder)
            else:
                self.word_embedding = tf.get_variable(
                    'word_embedding', [
                        self.vocab_size, self.embed_dim], initializer=self.initializer_emb)

            # projection with embeddings
            self.xfirst_sents = tf.nn.embedding_lookup(
                self.word_embedding, self.xfirst_sents)
            self.xlast_sents = tf.nn.embedding_lookup(
                self.word_embedding, self.xlast_sents)
            self.xmiddles_sents = tf.nn.embedding_lookup(
                self.word_embedding, self.xmiddles_sents)

            # INPUT DROPOUT
            if self.is_training and self.config.dropout_prob > 0:
                self.xfirst_sents = tf.nn.dropout(
                    self.xfirst_sents, keep_prob=1 - self.config.dropout_prob)
                self.xlast_sents = tf.nn.dropout(
                    self.xlast_sents, keep_prob=1 - self.config.dropout_prob)
            # inputs [ batch=128 x length=20 x hidden=512]
            self.xfirstlast_sents = tf.concat(
                [self.xfirst_sents, self.xlast_sents], 1)

        # Encoding first and last sentences
        with tf.variable_scope('encoder'):
            # https://stackoverflow.com/questions/39112622/how-do-i-set-tensorflow-rnn-state-when-state-is-tuple-true
            # Inititial state
            l = tf.unstack(self.initial_state, axis=0)
            initial_state_tuple = tuple([tf.nn.rnn_cell.LSTMStateTuple(
                l[idx][0], l[idx][1]) for idx in range(self.lstm_layers)])
            enc_output_first, enc_state_first = tf.nn.dynamic_rnn(
                self.cell_enc, self.xfirst_sents, sequence_length=self.xfirst_lens, dtype=self.dtype)
            # , initial_state=initial_state_tuple)
            enc_output_last, enc_state_last = tf.nn.dynamic_rnn(
                self.cell_enc, self.xlast_sents, sequence_length=self.xlast_lens, dtype=self.dtype)
            # , initial_state=initial_state_tuple)
            #enc_state = tf.concat([enc_state_first[0].h, enc_state_last[0].h], 1)
            enc_state = tuple([
                tf.nn.rnn_cell.LSTMStateTuple(
                    tf.concat([f[0], l[0]], 1), tf.concat([f[1], l[1]], 1))
                for f, l in zip(enc_state_first, enc_state_last)])
        return enc_state

    def decoder(self, enc_state):
        # Decoding first and last sentences
        states = [enc_state]
        with tf.variable_scope('decoder'):
            dec_outputs, dec_states = [], []
            for i in range(self.para_len - 2):
                dec_output, dec_state = tf.nn.dynamic_rnn(
                    self.cell_dec,
                    self.xmiddles_sents[i],
                    initial_state=states[-1],
                    sequence_length=self.xmiddles_lens[i])
                dec_outputs.append(tf.reshape(
                    tf.concat(dec_output, 1), [-1, self.lstm_size * 2]))
                dec_states.append(dec_state)
                states.append(dec_state)
        return dec_states, dec_outputs, states

    def loss_fn(self, dec_outputs):
        # LM Inference
        losses_lm = []
        with tf.variable_scope('loss_lm'):
            for i in range(self.para_len - 2):
                logit_lm = tf.matmul(dec_outputs[i], self.W_lm) + self.b_lm
                label_lm = tf.reshape(self.targets_sents[i], [-1])
                pred_lm = tf.nn.softmax(logit_lm)
                loss_lm = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=label_lm, logits=logit_lm)
                mask = tf.sequence_mask(
                    self.targets_lens[i], maxlen=self.step_size)
                mask = tf.reshape(mask, [-1])
                loss_lm_masked = tf.boolean_mask(loss_lm, mask)
                losses_lm.append(loss_lm)
            self.loss_lm = tf.add_n([tf.reduce_sum(l) for l in losses_lm])
        self.loss = self.loss_lm

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('loss_lm', self.loss_lm)
        self.summary_op = tf.summary.merge_all()

    def update_lr(self, session, learning_rate):
        if self.is_training:
            session.run(tf.assign(self.lr, learning_rate))

    def get_initial_state(self):
        return np.zeros([self.config.lstm_layers, 2, self.batch_size,
                         self.config.lstm_size], dtype=self.dtype)


class RSTSeq2Seq(Seq2Seq):
    def __init__(
            self,
            config,
            mode,
            pretrained_embeddings=None,
            device='/gpu:0',
            reuse=None):
        super().__init__(self, config, mode, pretrained_embeddings, device, reuse)
        self.name = 'RSTSeq2Seq'

        self.use_crf = self.config.use_crf  # True
        self.rst_size = self.config.rst_size
        self._sttrs += ['use_crf', 'rst_size']

        std_rst = np.sqrt(1. / self.rst_size)
        self.initializer_emb_rst = tf.random_uniform_initializer(
            -std_rst, std_rst)
        self.initializer_crf = tf.random_uniform_initializer(
            -std_rst * 0.8, std_rst * 0.8)

        self.W_crf = tf.get_variable(
            "W_crf", [
                self.lstm_size * 2, self.rst_size], initializer=self.initializer_crf)
        self.b_crf = tf.get_variable(
            "b_crf", [self.rst_size], initializer=self.initializer_constant)

    def init(self):
        print('Initializing...')
        with tf.device(self.device), tf.name_scope(self.mode), tf.variable_scope(self.name, reuse=self.reuse):
            self.placeholders()
            enc_state = self.encoder()
            dec_states, dec_outputs, states = self.decoder(enc_state)
            self.loss(dec_outputs)

            self.global_step = tf.Variable(
                0, name='global_step', trainable=False)
            if self.is_training:
                self.lr = tf.Variable(0.0, trainable=False)
                self.train_op = self.add_train_op(
                    'adagrad',
                    self.lr,
                    self.loss,
                    clip=self.config.max_grad_norm)
            else:
                self.train_op = tf.no_op()

    def placeholders(self):
        super().__init__(self)

        self.xmiddles_rsts = []
        self.targets_rsts = []
        for i in range(self.para_len - 2):
            self.xmiddles_rsts.append(
                tf.placeholder(
                    tf.int32, [
                        self.batch_size, self.step_infer_size], name='xmiddles_rsts%d' %
                    (i)))
            self.targets_rsts.append(
                tf.placeholder(
                    tf.int32, [
                        self.batch_size, self.step_infer_size], name='targets_rsts%d' %
                    (i)))

    def encoder(self):
        super().__init__(self)

    def loss(self, dec_outputs):

        # LM Inference
        super().__init__(self)

        # CRF Inference
        losses_crf = []
        with tf.variable_scope('loss_crf') as scope:
            for i in range(self.para_len - 2):
                if i > 0:
                    scope.reuse_variables()
                logit_crf = tf.matmul(dec_outputs[i], self.W_crf) + self.b_crf
                logit_crf = tf.reshape(
                    logit_crf, [
                        self.batch_size, self.step_size, -1])
                pred_crf = tf.nn.softmax(logit_crf)
                label_crf = self.targets_rsts[i]
                seq_crf = self.targets_lens[i]
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    logit_crf, label_crf, seq_crf)
                # self.trans_params = trans_params # need to evaluate it for
                # decoding
                losses_crf.append(-log_likelihood)
            self.loss_crf = tf.add_n([tf.reduce_sum(l) for l in losses_crf])

        # LM-CRF loss: J_lm + J_crf
        self.loss = tf.reduce_sum(self.loss_lm + self.loss_crf)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('loss_lm', self.loss_lm)
        tf.summary.scalar('loss_crf', self.loss_crf)
        self.summary_op = tf.summary.merge_all()
