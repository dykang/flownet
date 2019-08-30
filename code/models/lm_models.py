import os
import sys
import time
import numpy as np
import tensorflow as tf
from .base import Base


class RNN(Base):
    def __init__(
            self,
            config,
            mode,
            pretrained_embeddings=None,
            device='/gpu:0',
            reuse=None):

        self.config = config
        self.mode = mode
        self.name = 'RNN'

        if mode == "train":
            self.is_training = True
            self.batch_size = config.train_batch_size
            self.step_size = config.train_step_size
        elif mode == "valid":
            self.is_training = False
            self.batch_size = config.valid_batch_size
            self.step_size = config.valid_step_size
        elif mode == 'test':
            self.is_training = False
            self.batch_size = config.test_batch_size
            self.step_size = config.test_step_size
            if config.is_test:
                self.step_size = 1
        else:
            print('wrong', mode)
            sys.exit(1)

        self.lstm_forget_bias = config.lstm_forget_bias
        self.dtype = tf.float32

        self.update_embeddings = config.update_embeddings
        self.pretrained_embeddings = pretrained_embeddings

        self.dropout_prob = config.dropout_prob
        self.vocab_size = config.vocab_size
        self.embed_dim = config.word_embedding_dim
        self.lstm_size = config.lstm_size
        self.lstm_layers = config.lstm_layers

        self.adagrad_eps = config.adagrad_eps
        self.max_grad_norm = config.max_grad_norm

        self.para_len = config.para_len
        self.dataset = config.dataset
        self.reload = config.reload
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

        self.init_summary()

        self.std = np.sqrt(1. / self.vocab_size)
        self.init_emb = tf.random_uniform_initializer(-self.std, self.std)
        self.init_softmax = tf.random_uniform_initializer(
            -self.std * 0.8, self.std * 0.8)



        with tf.device(device), tf.name_scope(mode), tf.variable_scope(self.name, reuse=reuse):
            self.placeholder()
            self.embed()
            self.encoder()
            self.decoder()
            self.inference()

    # INPUTS and Targets
    def placeholder(self):
        self.xmiddles, self.targets, self.seq_lens = [], [], []
        for i in range(self.para_len):
            self.xmiddles.append(
                tf.placeholder(
                    tf.int32, [
                        self.batch_size, self.step_size], name='xmiddle%d' %
                    (i)))
            self.targets.append(
                tf.placeholder(
                    tf.int32, [
                        self.batch_size, self.step_size], name='target%d' %
                    (i)))
            self.seq_lens.append(
                tf.placeholder(
                    tf.int32, [
                        self.batch_size], name='seq_len%d' %
                    (i)))

        # Inititial state
        self.initial_state = tf.placeholder(
            tf.float32, [self.lstm_layers, 2, self.batch_size, self.lstm_size])

    # WORD EMBEDDING [batch x step] x [Vocab x emb_size]
    def embed(self):
        with tf.variable_scope('embed'):
            if self.pretrained_embeddings is not None:
                self.word_embedding = tf.get_variable(
                    'word_embedding', [self.vocab_size, self.embed_dim],
                    trainable=self.update_embeddings,
                    initializer=self.init_emb)
                self.embedding_placeholder = tf.placeholder(
                    tf.float32, [self.vocab_size, self.embed_dim])
                self.embedding_init = self.word_embedding.assign(
                    self.embedding_placeholder)
            else:
                self.word_embedding = tf.get_variable(
                    'word_embedding',
                    [self.vocab_size, self.embed_dim],
                    initializer=self.init_emb)

            # get [batch x step x embedding]
            xmiddles = tf.nn.embedding_lookup(
                self.word_embedding, self.xmiddles)

            # INPUT DROPOUT
            if self.is_training and self.dropout_prob > 0:
                xmiddles = tf.nn.dropout(
                    xmiddles, keep_prob=1 - self.dropout_prob)


    # RNN cell
    def encoder(self):
        with tf.variable_scope('encoder'):
            self.cell = tf.nn.rnn_cell.LSTMCell(
                self.lstm_size,
                forget_bias=self.lstm_forget_bias,
                state_is_tuple=True)
            if self.is_training and self.dropout_prob > 0:
                self.cell = tf.nn.rnn_cell.DropoutWrapper(
                    self.cell, output_keep_prob=1. - self.dropout_prob)
            self.cell = tf.nn.rnn_cell.MultiRNNCell(
                [self.cell] * self.lstm_layers, state_is_tuple=True)



    # Decoder
    def decoder(self):
        with tf.variable_scope('decoder'):

            # output layer
            W_lm = tf.get_variable(
                "W_lm", [
                    self.lstm_size, self.vocab_size], initializer=self.init_softmax)
            b_lm = tf.get_variable(
                "b_lm", [
                    self.vocab_size], initializer=tf.constant_initializer(
                    0.0, dtype=self.dtype))

            # initial state
            l = tf.unstack(self.initial_state, axis=0)
            state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
                l[idx][0], l[idx][1]) for idx in range(self.lstm_layers)])

            losses, logits, preds, dec_states, costs = [], [], [], [], []
            for i in range(self.para_len):
                dec_output, state = tf.nn.dynamic_rnn(
                    self.cell, self.xmiddles[i],
                    initial_state=state, sequence_length=self.seq_lens[i])
                dec_states.append(state)
                output = tf.reshape(
                    tf.concat(dec_output, 1), [-1, self.lstm_size])
                label = tf.reshape(self.targets[i], [-1])
                logit = tf.matmul(output, W_lm) + b_lm
                logits.append(logit)
                # preds.append(tf.nn.softmax(logit))

                # masking using sequence lengths (not including padding
                # symbols)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=label, logits=logit)
                mask = tf.reshape(tf.sequence_mask(
                    self.seq_lens[i], self.step_size), [-1])
                losses.append(tf.reduce_sum(tf.boolean_mask(loss, mask)))

        #self.cost = tf.add_n([tf.reduce_sum(l) for l in losses])
        self.loss = tf.reduce_sum(losses)


    # Inference
    def inference(self):
        self.global_step = tf.Variable(
            0, name='global_step', trainable=False)

        self.summary_op = tf.summary.scalar(
            "loss_%s" % (self.mode), self.loss)
        #self.summary_op = tf.summary.merge_all()

        if self.is_training:
            self.lr = tf.Variable(0.0, trainable=False)
            self.train_op = self.add_train_op(
                'adam', self.lr, self.loss, clip=config.max_grad_norm)
        else:
            self.train_op = tf.no_op()

    def update_lr(self, session, learning_rate):
        if self.is_training:
            session.run(tf.assign(self.lr, learning_rate))

    def get_initial_state(self):
        return np.zeros([self.lstm_layers, 2, self.batch_size,
                         self.lstm_size], dtype=self.dtype)



class HRNN(RNN):
    def __init__(
            self,
            config,
            mode,
            pretrained_embeddings=None,
            device='/gpu:0',
            reuse=None):

        self.name = 'HRNN'

        super().__init__(config, mode, pretrained_embeddings, device, reuse)

        with tf.device(device), tf.name_scope(mode), tf.variable_scope(self.name, reuse=reuse):
            self.placeholder()
            self.embed()
            self.encoder()
            self.decoder()
            self.inference()

    def encoder(self):
        with tf.variable_scope('encoder'):
            # Sentence Encoding
            with tf.device(device), tf.name_scope(mode), tf.variable_scope("word_enc", reuse=reuse):
                cell = tf.nn.rnn_cell.LSTMCell(
                    self.lstm_size,
                    forget_bias=self.lstm_forget_bias,
                    state_is_tuple=True)
                if self.is_training and self.dropout_prob > 0:
                    cell = tf.nn.rnn_cell.DropoutWrapper(
                        cell, output_keep_prob=1. - self.dropout_prob)
                self.cell = tf.nn.rnn_cell.MultiRNNCell(
                    [cell] * self.lstm_layers, state_is_tuple=True)

            with tf.device(device), tf.name_scope(mode), tf.variable_scope("sent_enc", reuse=reuse):
                sent_cell = tf.nn.rnn_cell.LSTMCell(
                    self.lstm_size, forget_bias=self.lstm_forget_bias, state_is_tuple=True)
                if self.is_training and self.dropout_prob > 0:
                    sent_cell = tf.nn.rnn_cell.DropoutWrapper(
                        sent_cell, output_keep_prob=1. - self.dropout_prob)
                self.sent_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [sent_cell] * self.lstm_layers, state_is_tuple=True)

            # Document Encoding NOTE document LSTM size == word embedding size
            with tf.device(device), tf.name_scope(mode), tf.variable_scope("sent_dec", reuse=reuse):
                doc_cell = tf.nn.rnn_cell.LSTMCell(
                    self.lstm_size + self.embed_dim, state_is_tuple=True)
                if self.is_training and self.dropout_prob > 0:
                    doc_cell = tf.nn.rnn_cell.DropoutWrapper(
                        doc_cell, output_keep_prob=1. - self.dropout_prob)
                self.doc_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [doc_cell] * self.lstm_layers, state_is_tuple=True)


    def decoder(self):
        with tf.variable_scope('decoder'):
            W_lm = tf.get_variable("W_lm", [self.lstm_size, self.vocab_size],
                                   initializer=self.init_softmax)
            b_lm = tf.get_variable(
                "b_lm", [
                    self.vocab_size], initializer=tf.constant_initializer(
                    0.0, dtype=self.dtype))

            # (1) word-level word en(de)coding
            with tf.device(device), tf.name_scope(mode), tf.variable_scope("word_encoding", reuse=reuse):
                l = tf.unstack(self.initial_state, axis=0)
                state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
                    l[idx][0], l[idx][1]) for idx in range(self.lstm_layers)])
                word_enc_outputs = []
                word_enc_states = []
                for i in range(self.para_len):
                    # TODO no use state again from previous sentence, get random
                    word_enc_output, word_enc_state = tf.nn.dynamic_rnn(
                        cell, xmiddles[i], initial_state=state, sequence_length=self.seq_lens[i])
                    word_enc_outputs.append(word_enc_output)
                    word_enc_states.append(word_enc_state)

            # (1.2) sentence-level word encoding
            with tf.device(device), tf.name_scope(mode), tf.variable_scope("sent_encoding", reuse=reuse):
                l = tf.unstack(self.get_initial_state(), axis=0)
                state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
                    l[idx][0], l[idx][1]) for idx in range(self.lstm_layers)])
                sent_states = tf.transpose(
                    tf.stack([h[:, -1, :] for h in word_enc_outputs]), [1, 0, 2])
                sent_enc_outputs = []
                sent_enc_output, state = tf.nn.dynamic_rnn(
                    sent_cell, sent_states, initial_state=state)
                sent_enc_outputs.append(sent_enc_output)

            # (2) sentence-level word decoding
            with tf.device(device), tf.name_scope(mode), tf.variable_scope("wordsent_decoding", reuse=reuse):

                l = tf.unstack(self.get_initial_sent_state(), axis=0)
                state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
                    l[idx][0], l[idx][1]) for idx in range(self.lstm_layers)])

                second_losses, second_logits = [], []
                for i in range(self.para_len):

                    # combnie word representation + sentence representation
                    # [batch x step_size x (embedding:300 + doc_state:512)]
                    if i == 0:
                        sent_enc = np.zeros(
                            [self.batch_size, self.lstm_size], dtype=self.dtype)
                    else:
                        sent_enc = sent_enc_output[:, i - 1, :]

                    sent_enc_output = tf.reshape(
                        tf.tile(
                            sent_enc, [
                                self.step_size, 1]), [
                            self.batch_size, self.step_size, -1])
                    xmiddle = tf.concat([xmiddles[i], sent_enc_output], 2)

                    dec_output, state = tf.nn.dynamic_rnn(
                        doc_cell, xmiddle, initial_state=state, sequence_length=self.seq_lens[i])

                    output = tf.reshape(tf.concat(
                        dec_output, 1), [-1, self.lstm_size + self.embed_dim])
                    label = tf.reshape(self.targets[i], [-1])
                    logit = tf.matmul(output, W_lm) + b_lm
                    second_logits.append(logit)

                    # masking using sequence lengths (not including padding
                    # symbols)
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=label, logits=logit)
                    mask = tf.reshape(tf.sequence_mask(
                        self.seq_lens[i], self.step_size), [-1])
                    second_losses.append(
                        tf.reduce_sum(
                            tf.boolean_mask(
                                loss, mask)))

            self.sent_loss_rate = tf.Variable(
                0.0, trainable=False, dtype=self.dtype)
            self.new_sent_loss_rate = tf.placeholder(
                tf.float32, shape=[], name="new_sent_loss_rate")
            self.sent_loss_rate_update = tf.assign(
                self.sent_loss_rate, self.new_sent_loss_rate)

            #self.cost = tf.reduce_sum(first_losses) + self.sent_loss_rate *    tf.reduce_sum(second_losses)
            self.loss = tf.reduce_sum(second_losses)





class HRED(Base):
    def __init__(
            self,
            config,
            mode,
            pretrained_embeddings=None,
            device='/gpu:0',
            reuse=None):

        self.mode = mode
        self.name = 'HRNN'

        if mode == "train":
            self.is_training = True
            self.batch_size = config.train_batch_size
            self.step_size = config.train_step_size
        elif mode == "valid":
            self.is_training = False
            self.batch_size = config.valid_batch_size
            self.step_size = config.valid_step_size
        elif mode == 'test':
            self.is_training = False
            self.batch_size = config.test_batch_size
            self.step_size = config.test_step_size
            if config.is_test:
                self.step_size = 1
        else:
            print('wrong', mode)
            sys.exit(1)

        lstm_forget_bias = config.lstm_forget_bias
        dtype = tf.float32

        self.update_embeddings = config.update_embeddings
        self.pretrained_embeddings = pretrained_embeddings

        self.dropout_prob = config.dropout_prob
        self.vocab_size = config.vocab_size
        self.embed_dim = config.word_embedding_dim
        self.lstm_size = config.lstm_size
        self.lstm_layers = config.lstm_layers

        self.adagrad_eps = config.adagrad_eps
        self.max_grad_norm = config.max_grad_norm

        self.para_len = config.para_len
        self.dataset = config.dataset
        self.reload = config.reload
        self.checkpoint_dir = "/data/tf/ckpts/"
        self.log_dir = "/data/tf/logs/"

        self._attrs = [
            'batch_size',
            'vocab_size',
            'step_size',
            'embed_dim',
            'lstm_size',
            'lstm_layers']

        self.std = np.sqrt(1. / self.vocab_size)
        self.init_emb = tf.random_uniform_initializer(-self.std, self.std)
        self.init_softmax = tf.random_uniform_initializer(
            -self.std * 0.8, self.std * 0.8)

        with tf.device(device), tf.name_scope(mode), tf.variable_scope("LSTMLM", reuse=reuse):

            # INPUTS and TARGETS
            self.xmiddles, self.targets, self.seq_lens = [], [], []
            for i in range(self.para_len):
                self.xmiddles.append(
                    tf.placeholder(
                        tf.int32, [
                            self.batch_size, self.step_size], name='xmiddle%d' %
                        (i)))
                self.targets.append(
                    tf.placeholder(
                        tf.int32, [
                            self.batch_size, self.step_size], name='target%d' %
                        (i)))
                self.seq_lens.append(
                    tf.placeholder(
                        tf.int32, [
                            self.batch_size], name='seq_len%d' %
                        (i)))

            # Inititial state
            self.initial_state = tf.placeholder(
                tf.float32, [self.lstm_layers, 2, self.batch_size, self.lstm_size])

            # Use pre-trained embedding
            # WORD EMBEDDING [batch x step] x [Vocab x emb_size]
            if self.pretrained_embeddings is not None:
                self.word_embedding = tf.get_variable(
                    'word_embedding', [self.vocab_size, self.embed_dim],
                    trainable=self.update_embeddings,
                    initializer=self.init_emb)
                self.embedding_placeholder = tf.placeholder(
                    tf.float32, [self.vocab_size, self.embed_dim])
                self.embedding_init = self.word_embedding.assign(
                    self.embedding_placeholder)
            else:
                self.word_embedding = tf.get_variable(
                    'word_embedding',
                    [self.vocab_size, self.embed_dim],
                    initializer=self.init_emb)

            # get [batch x step x embeddsng]
            xmiddles = tf.nn.embedding_lookup(
                self.word_embedding, self.xmiddles)

            # INPUT DROPOUT
            if self.is_training and self.dropout_prob > 0:
                xmiddles = tf.nn.dropout(
                    xmiddles, keep_prob=1 - self.dropout_prob)

            # Decoding
            W_lm = tf.get_variable("W_lm",
                                   [self.lstm_size + self.embed_dim,
                                    self.vocab_size],
                                   initializer=self.init_softmax)
            b_lm = tf.get_variable(
                "b_lm", [
                    self.vocab_size], initializer=tf.constant_initializer(
                    0.0, dtype=tf.float32))

            # Sentence Encoding
            cell = tf.nn.rnn_cell.LSTMCell(
                self.lstm_size,
                forget_bias=self.lstm_forget_bias,
                state_is_tuple=True)
            if self.is_training and self.dropout_prob > 0:
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=1. - self.dropout_prob)
            enc_cell = tf.nn.rnn_cell.MultiRNNCell(
                [cell] * self.lstm_layers, state_is_tuple=True)

            # Document Encoding NOTE document LSTM size == word embedding size
            doc_cell = tf.nn.rnn_cell.LSTMCell(
                self.lstm_size + self.embed_dim, state_is_tuple=True)
            if self.is_training and self.dropout_prob > 0:
                doc_cell = tf.nn.rnn_cell.DropoutWrapper(
                    doc_cell, output_keep_prob=1. - self.dropout_prob)
            dec_cell = tf.nn.rnn_cell.MultiRNNCell(
                [doc_cell] * self.lstm_layers, state_is_tuple=True)

            # initial state
            # inputs [ batch=128 x length=20 x hidden=512]
            # https://stackoverflow.com/questions/39112622/how-do-i-set-tensorflow-rnn-state-when-state-is-tuple-true
            l = tf.unstack(self.initial_state, axis=0)
            initial_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
                l[idx][0], l[idx][1]) for idx in range(self.lstm_layers)])

            sent_initial_state = self.get_initial_sent_state()
            l = tf.unstack(sent_initial_state, axis=0)
            sent_initial_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
                l[idx][0], l[idx][1]) for idx in range(self.lstm_layers)])

            with tf.device(device), tf.name_scope(mode), tf.variable_scope("hred", reuse=reuse):
                losses, logits, preds, dec_states, costs = [], [], [], [], []

                # Sentence decoding
                sent_enc_states = []
                dec_outputs = []

                enc_state = initial_state
                dec_state = sent_initial_state
                for i in range(1, self.para_len):

                    # sentence encoding
                    with tf.device(device), tf.name_scope(mode), tf.variable_scope("sent_enc", reuse=reuse):
                        enc_output, enc_state = tf.nn.dynamic_rnn(
                            enc_cell, xmiddles[i - 1], initial_state=enc_state, sequence_length=self.seq_lens[i - 1])

                    # sentence decoding
                    with tf.device(device), tf.name_scope(mode), tf.variable_scope("doc_enc", reuse=reuse):

                        # NOTE between VHRED and HRED
                        # [batch x step_size x (embedding:300 + doc_state:512)]
                        xmiddle = tf.concat([xmiddles[i], enc_output], 2)
                        seq_len = self.seq_lens[i]

                        dec_output, dec_state = tf.nn.dynamic_rnn(
                            dec_cell, xmiddle, initial_state=dec_state, sequence_length=seq_len)
                        output = tf.reshape(tf.concat(
                            dec_output, 1), [-1, self.lstm_size + self.embed_dim])
                        label = tf.reshape(self.targets[i], [-1])
                        logit = tf.matmul(output, W_lm) + b_lm
                        logits.append(logit)
                        # preds.append(tf.nn.softmax(logit))

                        # masking using sequence lengths (not including padding
                        # symbols)
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=label, logits=logit)
                        mask = tf.reshape(tf.sequence_mask(
                            self.seq_lens[i], self.step_size), [-1])
                        losses.append(
                            tf.reduce_sum(
                                tf.boolean_mask(
                                    loss, mask)))

            #self.cost = tf.add_n([tf.reduce_sum(l) for l in losses])
            self.cost = tf.reduce_sum(losses)
            self.global_step = tf.Variable(
                0, name='global_step', trainable=False)

            if self.is_training:
                self.lr = tf.Variable(0.0, trainable=False)
                optimizer = tf.train.AdagradOptimizer(self.lr, self.adagrad_eps)
                tvars = tf.trainable_variables()
                grads = tf.gradients(
                    [self.cost / self.batch_size], tvars)
                grads = [
                    tf.clip_by_norm(
                        grad,
                        self.max_grad_norm) if grad is not None else grad for grad in grads]
                self.eval_op = optimizer.apply_gradients(zip(grads, tvars))
            else:
                self.eval_op = tf.no_op()

    def update_lr(self, session, learning_rate):
        if self.is_training:
            session.run(tf.assign(self.lr, learning_rate))

    def get_initial_state(self):
        return np.zeros([self.lstm_layers, 2, self.batch_size,
                         self.lstm_size], dtype=np.float32)

    def get_initial_sent_state(self):
        return np.zeros([self.lstm_layers, 2, self.batch_size,
                         self.lstm_size + self.embed_dim], dtype=np.float32)


class DHRNN(Base):
    def __init__(
            self,
            config,
            mode,
            pretrained_embeddings=None,
            device='/gpu:0',
            reuse=None):

        self.mode = mode
        self.name = 'DHRNN'

        if mode == "train":
            self.is_training = True
            self.batch_size = config.train_batch_size
            self.step_size = config.train_step_size
        elif mode == "valid":
            self.is_training = False
            self.batch_size = config.valid_batch_size
            self.step_size = config.valid_step_size
        elif mode == 'test':
            self.is_training = False
            self.batch_size = config.test_batch_size
            self.step_size = config.test_step_size
            if config.is_test:
                self.step_size = 1
        else:
            print('wrong', mode)
            sys.exit(1)

        lstm_forget_bias = config.lstm_forget_bias
        dtype = tf.float32

        self.update_embeddings = config.update_embeddings
        self.pretrained_embeddings = pretrained_embeddings

        self.dropout_prob = config.dropout_prob
        self.vocab_size = config.vocab_size
        self.embed_dim = config.word_embedding_dim
        self.lstm_size = config.lstm_size
        self.lstm_layers = config.lstm_layers

        self.adagrad_eps = config.adagrad_eps
        self.max_grad_norm = config.max_grad_norm

        self.para_len = config.para_len
        self.dataset = config.dataset
        self.reload = config.reload
        self.checkpoint_dir = "/data/tf/ckpts/"
        self.log_dir = "/data/tf/logs/"

        self._attrs = [
            'batch_size',
            'vocab_size',
            'step_size',
            'embed_dim',
            'lstm_size',
            'lstm_layers']

        self.std = np.sqrt(1. / self.vocab_size)
        self.init_emb = tf.random_uniform_initializer(-self.std, self.std)
        self.init_softmax = tf.random_uniform_initializer(
            -self.std * 0.8, self.std * 0.8)

        with tf.device(device), tf.name_scope(mode), tf.variable_scope(self.name, reuse=reuse):

            # INPUTS and TARGETS
            self.xmiddles, self.targets, self.seq_lens = [], [], []
            for i in range(self.para_len):
                self.xmiddles.append(
                    tf.placeholder(
                        tf.int32, [
                            self.batch_size, self.step_size], name='xmiddle%d' %
                        (i)))
                self.targets.append(
                    tf.placeholder(
                        tf.int32, [
                            self.batch_size, self.step_size], name='target%d' %
                        (i)))
                self.seq_lens.append(
                    tf.placeholder(
                        tf.int32, [
                            self.batch_size], name='seq_len%d' %
                        (i)))

            # Inititial state
            self.initial_state = tf.placeholder(
                tf.float32, [self.lstm_layers, 2, self.batch_size, self.lstm_size])

            # Use pre-trained embedding
            # WORD EMBEDDING [batch x step] x [Vocab x emb_size]
            if self.pretrained_embeddings is not None:
                self.word_embedding = tf.get_variable(
                    'word_embedding', [self.vocab_size, self.embed_dim],
                    trainable=self.update_embeddings,
                    initializer=self.init_emb)
                self.embedding_placeholder = tf.placeholder(
                    tf.float32, [self.vocab_size, self.embed_dim])
                self.embedding_init = self.word_embedding.assign(
                    self.embedding_placeholder)
            else:
                self.word_embedding = tf.get_variable(
                    'word_embedding',
                    [self.vocab_size, self.embed_dim],
                    initializer=self.init_emb)

            # get [batch x step x embedding]
            xmiddles = tf.nn.embedding_lookup(
                self.word_embedding, self.xmiddles)

            # INPUT DROPOUT
            if self.is_training and self.dropout_prob > 0:
                xmiddles = tf.nn.dropout(
                    xmiddles, keep_prob=1 - self.dropout_prob)

            W_lm = tf.get_variable("W_lm", [self.lstm_size, self.vocab_size],
                                   initializer=self.init_softmax)
            b_lm = tf.get_variable(
                "b_lm", [
                    self.vocab_size], initializer=tf.constant_initializer(
                    0.0, dtype=tf.float32))

            W_lm2 = tf.get_variable("W_lm2",
                                    [self.lstm_size + self.embed_dim,
                                     self.vocab_size],
                                    initializer=self.init_softmax)
            b_lm2 = tf.get_variable(
                "b_lm2", [
                    self.vocab_size], initializer=tf.constant_initializer(
                    0.0, dtype=tf.float32))

            # Word Encoding
            with tf.device(device), tf.name_scope(mode), tf.variable_scope("word_enc", reuse=reuse):
                cell = tf.nn.rnn_cell.LSTMCell(
                    self.lstm_size,
                    forget_bias=self.lstm_forget_bias,
                    state_is_tuple=True)
                if self.is_training and self.dropout_prob > 0:
                    cell = tf.nn.rnn_cell.DropoutWrapper(
                        cell, output_keep_prob=1. - self.dropout_prob)
                cell = tf.nn.rnn_cell.MultiRNNCell(
                    [cell] * self.lstm_layers, state_is_tuple=True)

            # Sentence Encoding
            with tf.device(device), tf.name_scope(mode), tf.variable_scope("sent_enc", reuse=reuse):
                sent_cell = tf.nn.rnn_cell.LSTMCell(
                    self.lstm_size, forget_bias=self.lstm_forget_bias, state_is_tuple=True)
                if self.is_training and self.dropout_prob > 0:
                    sent_cell = tf.nn.rnn_cell.DropoutWrapper(
                        sent_cell, output_keep_prob=1. - self.dropout_prob)
                sent_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [sent_cell] * self.lstm_layers, state_is_tuple=True)

            # Delta Encoding
            with tf.device(device), tf.name_scope(mode), tf.variable_scope("delta_enc", reuse=reuse):
                delta_cell = tf.nn.rnn_cell.LSTMCell(
                    self.lstm_size, state_is_tuple=True)
                if self.is_training and self.dropout_prob > 0:
                    delta_cell = tf.nn.rnn_cell.DropoutWrapper(
                        delta_cell, output_keep_prob=1. - self.dropout_prob)
                delta_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [delta_cell] * self.lstm_layers, state_is_tuple=True)

            use_delta_only = True

            # Document decoding NOTE document LSTM size == word + sentence +
            # delta
            with tf.device(device), tf.name_scope(mode), tf.variable_scope("sent_dec", reuse=reuse):

                if use_delta_only:
                    doc_cell = tf.nn.rnn_cell.LSTMCell(
                        self.lstm_size + self.embed_dim, state_is_tuple=True)
                else:
                    doc_cell = tf.nn.rnn_cell.LSTMCell(
                        self.lstm_size + self.lstm_size + self.embed_dim, state_is_tuple=True)

                if self.is_training and self.dropout_prob > 0:
                    doc_cell = tf.nn.rnn_cell.DropoutWrapper(
                        doc_cell, output_keep_prob=1. - self.dropout_prob)
                doc_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [doc_cell] * self.lstm_layers, state_is_tuple=True)

            # (1.1) word-level word en(de)coding
            with tf.device(device), tf.name_scope(mode), tf.variable_scope("word_encoding", reuse=reuse):
                l = tf.unstack(self.initial_state, axis=0)
                state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
                    l[idx][0], l[idx][1]) for idx in range(self.lstm_layers)])

                word_enc_outputs, logits = [], []
                first_losses = []
                word_enc_states = []
                for i in range(self.para_len):
                    word_enc_output, state = tf.nn.dynamic_rnn(
                        cell, xmiddles[i], initial_state=state, sequence_length=self.seq_lens[i])
                    word_enc_outputs.append(word_enc_output)
                    word_enc_states.append(state)

            # (1.2) sentence-level word encoding
            with tf.device(device), tf.name_scope(mode), tf.variable_scope("sent_encoding", reuse=reuse):
                l = tf.unstack(self.get_initial_state(), axis=0)
                state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
                    l[idx][0], l[idx][1]) for idx in range(self.lstm_layers)])

                sent_states = tf.transpose(
                    tf.stack([h[:, -1, :] for h in word_enc_outputs]), [1, 0, 2])
                sent_enc_output, state = tf.nn.dynamic_rnn(
                    sent_cell, sent_states, initial_state=state)

            # (1.3) delta-level word encoding
            with tf.device(device), tf.name_scope(mode), tf.variable_scope("delta_encoding", reuse=reuse):
                l = tf.unstack(self.get_initial_state(), axis=0)
                state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
                    l[idx][0], l[idx][1]) for idx in range(self.lstm_layers)])

                delta_encs = []
                for i in range(self.para_len):
                    if i == 0:
                        prev_sent_enc = np.zeros(
                            [self.batch_size, self.lstm_size], dtype=np.float32)
                    else:
                        prev_sent_enc = sent_enc_output[:, i - 1, :]
                    current_sent_enc = sent_enc_output[:, i, :]

                    delta_enc = tf.subtract(current_sent_enc, prev_sent_enc)
                    delta_encs.append(delta_enc)

                # model deltas using RNN or not
                delta_states = tf.transpose(
                    tf.stack([d for d in delta_encs]), [1, 0, 2])

                delta_enc_output, state = tf.nn.dynamic_rnn(
                    delta_cell, delta_states, initial_state=state)


            # (2) sentence-level word decoding
            with tf.device(device), tf.name_scope(mode), tf.variable_scope("wordsent_decoding", reuse=reuse):

                l = tf.unstack(self.get_initial_sent_state(), axis=0)
                state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
                    l[idx][0], l[idx][1]) for idx in range(self.lstm_layers)])

                second_losses, second_logits = [], []
                for i in range(self.para_len):

                    if use_delta_only:
                        enc_output = delta_enc_output
                    else:
                        enc_output = tf.concat(
                            [sent_enc_output, delta_enc_output], 2)

                    # combnie word representation + sentence representation
                    # xmiddle = tf.concat([xmiddles[i],word_enc_outputs[i]],2) #
                    # [batch x step_size x (embedding:300 + doc_state:512)]
                    if i == 0:
                        sent_enc = np.zeros(
                            [self.batch_size, self.lstm_size], dtype=np.float32)
                    else:
                        sent_enc = enc_output[:, i - 1, :]

                    enc_output = tf.reshape(
                        tf.tile(
                            sent_enc, [
                                self.step_size, 1]), [
                            self.batch_size, self.step_size, -1])
                    xmiddle = tf.concat([xmiddles[i], enc_output], 2)

                    dec_output, state = tf.nn.dynamic_rnn(
                        doc_cell, xmiddle, initial_state=state, sequence_length=self.seq_lens[i])

                    output = tf.reshape(tf.concat(
                        dec_output, 1), [-1, self.lstm_size + self.embed_dim])
                    label = tf.reshape(self.targets[i], [-1])
                    logit = tf.matmul(output, W_lm2) + b_lm2
                    second_logits.append(logit)

                    # masking using sequence lengths (not including padding
                    # symbols)
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=label, logits=logit)
                    mask = tf.reshape(tf.sequence_mask(
                        self.seq_lens[i], self.step_size), [-1])
                    second_losses.append(
                        tf.reduce_sum(
                            tf.boolean_mask(
                                loss, mask)))

            self.sent_loss_rate = tf.Variable(
                0.0, trainable=False, dtype=tf.float32)
            self.new_sent_loss_rate = tf.placeholder(
                tf.float32, shape=[], name="new_sent_loss_rate")
            self.sent_loss_rate_update = tf.assign(
                self.sent_loss_rate, self.new_sent_loss_rate)

            #self.cost = tf.reduce_sum(first_losses) + self.sent_loss_rate *    tf.reduce_sum(second_losses)
            self.cost = tf.reduce_sum(second_losses)
            self.global_step = tf.Variable(
                0, name='global_step', trainable=False)

            if self.is_training:
                self.lr = tf.Variable(0.0, trainable=False)
                optimizer = tf.train.AdagradOptimizer(self.lr, self.adagrad_eps)
                tvars = tf.trainable_variables()
                grads = tf.gradients(
                    [self.cost / self.batch_size], tvars)
                grads = [
                    tf.clip_by_norm(
                        grad,
                        self.max_grad_norm) if grad is not None else grad for grad in grads]
                self.eval_op = optimizer.apply_gradients(zip(grads, tvars))
            else:
                self.eval_op = tf.no_op()

    def update_lr(self, session, learning_rate):
        if self.is_training:
            session.run(tf.assign(self.lr, learning_rate))

    def get_initial_state(self):
        return np.zeros([self.lstm_layers, 2, self.batch_size,
                         self.lstm_size], dtype=np.float32)

    def get_initial_sent_state(self):
        return np.zeros([self.lstm_layers, 2, self.batch_size,
                         self.lstm_size + self.embed_dim], dtype=np.float32)
