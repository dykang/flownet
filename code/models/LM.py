import numpy as np
import os
import sys
import random
import io
import time
import math
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from prepare import data_utils
from models.base import Model
from utils import *

logging = tf.logging


class LM(Model):
    def __init__(
            self,
            sess,
            reader,
            batch_size=100,
            rnn_size=512,
            layer_depth=1,
            word_emb_dim=512,
            max_len=50,
            h_dim=50,
            word_vocab_size=10000,
            learning_rate=0.001,
            decay_rate=0.96,
            dropout_prob=0.5,
            max_epochs=1000,
            max_max_epochs=100,
            max_gradient_norm=0.5,
            annealing_pivot=-1,
            checkpoint_dir="checkpoint",
            log_dir="logs",
            epochs_per_checkpoint=50,
            forward_only=False,
            dataset_dir="data",
            dataset="ptb",
            use_progressbar=False,
            buckets=None,
            max_train_data_size=0,
            optimizer='sgd',
            anneal_type=None,
            const=None,
            autoenc=False):

        self.sess = sess
        self.reader = reader
        self.rnn_size = rnn_size
        self.layer_depth = layer_depth
        self.word_emb_dim = word_emb_dim
        self.max_len = max_len
        self.word_vocab_size = word_vocab_size
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.max_epochs = max_epochs
        self.max_max_epochs = max_max_epochs
        self.dropout_prob = dropout_prob
        self.forward_only = forward_only
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.epochs_per_checkpoint = epochs_per_checkpoint
        self.dataset_dir = dataset_dir
        self.dataset = dataset
        self.use_progressbar = use_progressbar
        self.optimizer = optimizer

        self.dtype = tf.float32
        self._attrs = [
            "word_emb_dim",
            "rnn_size",
            "layer_depth",
            "decay_rate",
            "dropout_prob",
            "use_batch_norm",
            "optimizer"]

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * self.decay_rate)

        if buckets:
            self._buckets = buckets
        else:
            self._buckets = [15, 20, 30, 40, 50]  # , (70, 80), (180, 198)]
        print('Bucket used: ', self._buckets)

        train_file, valid_file, test_file, vocab_file = self.reader.prepare_data()
        self.train_set = self.read_data(train_file, self._buckets)
        # , max_size = max_train_data_size)
        self.valid_set = self.read_data(valid_file, self._buckets)
        self.test_set = self.read_data(test_file, self._buckets)
        self.word2idx, self.idx2word = data_utils.initialize_vocabulary(
            vocab_file)
        print('Size of vocab: ', len(self.word2idx), len(self.idx2word))

        self.test_file = test_file
        self.vocab_file = vocab_file

    def read_data(self, data_path, _buckets, max_size=None):
        print('Load data from "{}"'.format(data_path))
        data_set = [[] for _ in _buckets]
        with io.open(data_path, encoding='utf8') as source_file:
            counter = 0
            while not max_size or counter < max_size:
                source_ids = source_file.readline()
                source_ids = source_ids.strip().split()
                if not source_ids:
                    break
                counter += 1
                if not source_ids:
                    continue
                for bucket_id, seq_length in enumerate(_buckets):
                    if len(source_ids) < seq_length:
                        data_set[bucket_id].append(source_ids)
                        break
        print('Buckets:', ' '.join([str(len(data_set[i]))
                                    for i, b in enumerate(_buckets)]))
        print('total bucket size = %d out of %d' %
              (sum(map(len, data_set)), counter))
        return data_set

    def create_model(self):
        self.encoder_inputs = []
        self.target_weights = []
        max_encoder_size = self._buckets[-1]  # , self._buckets[-1] + 1
        for i in xrange(max_encoder_size + 1):
            self.encoder_inputs.append(tf.placeholder(
                tf.int32, shape=[None],
                name='encoder{0}'.format(i)))
            self.target_weights.append(tf.placeholder(
                tf.float32, shape=[None], name='weight{0}'.format(i)))

        self.targets = [self.encoder_inputs[i + 1]
                        for i in xrange(len(self.encoder_inputs) - 1)]

        self.embedding = tf.get_variable(
            'embedding', [
                self.word_vocab_size, self.word_emb_dim], trainable=True)
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [cell] * self.layer_depth, state_is_tuple=True)
        self.cell = cell

        def rnnlm(encoder_inputs, targets, weights):

            emb_encoder_inputs = [tf.nn.embedding_lookup(self.embedding, i)
                                  for i in encoder_inputs]

            self.initial_state = self.cell.zero_state(
                self.batch_size, tf.float32)
            proj_w = tf.get_variable(
                'proj_w', [
                    self.rnn_size, self.word_vocab_size])
            proj_b = tf.get_variable('proj_b', [self.word_vocab_size])
            if self.forward_only:
                loop_function = extract_argmax_and_embed(
                    self.embedding, (proj_w, proj_b),
                    update_embedding=False)
            else:
                loop_function = None
            outputs, _ = tf.nn.seq2seq.rnn_decoder(
                emb_encoder_inputs,
                self.initial_state,
                self.cell,
                loop_function=loop_function,
                scope='rnn_decoder')

            logits = [tf.nn.xw_plus_b(output, proj_w, proj_b)
                      for output in outputs]

            # cross entropy loss = -sum(y * log(p(y))
            loss = tf.nn.seq2seq.sequence_loss(logits, targets, weights)
            return logits, loss

        self.losses = []
        self.outputs = []
        for j, seq_length in enumerate(self._buckets):
            with tf.variable_scope(tf.get_variable_scope(),
                                   reuse=True if j > 0 else None):
                bucket_outputs, loss = rnnlm(
                    self.encoder_inputs[:seq_length],
                    self.targets[:seq_length],
                    self.target_weights[:seq_length])
                self.outputs.append(bucket_outputs)
                self.losses.append(loss)

        self.updates = []
        self.gradient_norms = []
        if self.optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer == 'adam':
            opt = tf.train.AdamOptimizer()
        else:
            print('wrong opt', self.optimizer)
            sys.exit(1)

        params = tf.trainable_variables()
        if not self.forward_only:
            for b in xrange(len(self._buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(
                    gradients, self.max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params),
                    global_step=self.global_step))
        self.initialize(log_dir=self.log_dir)

    def train(self):

        print('training...')
        self.create_model()

        train_bucket_sizes = [len(self.train_set[b])
                              for b in xrange(len(self._buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        step_time, loss = 0.0, 0.0
        previous_losses = []

        for epoch in range(self.max_epochs):
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > np.random.random_sample()])
            start_time = time.time()
            encoder_inputs, target_weights = self.get_batch(
                self.train_set, bucket_id)
            #print (np.array(encoder_inputs).shape,np.array(target_weights))
            #print (' '.join([str(n) for n in np.array(encoder_inputs)[:,-1]] ))
            #print (' '.join([self.idx2word[n] for n in np.array(encoder_inputs)[:,-1]] ))
            # sys.exit(1)

            norm, step_loss = self.step(
                encoder_inputs, target_weights, bucket_id)
            step_time += (time.time() - start_time) / self.epochs_per_checkpoint
            loss += step_loss / self.epochs_per_checkpoint
            # print ("epoch [%2d]: %2.4f"%(epoch, step_loss)) #,
            # np.exp(step_loss)))

            if epoch % self.epochs_per_checkpoint == 0 and epoch > 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print(
                    "global step %d learning rate %.4f step-time %.2f perplexity %.2f" %
                    (self.global_step.eval(), self.learning_rate.eval(), step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3
                # times.
                if len(previous_losses) > 2 and loss > max(
                        previous_losses[-3:]):
                    self.sess.run(self.learning_rate_decay_op)
                previous_losses.append(loss)

                step_time, loss = 0.0, 0.0

                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(self._buckets)):
                    encoder_inputs, target_weights = self.get_batch(
                        self.valid_set, bucket_id)
                    _, eval_loss, _ = self.step(
                        encoder_inputs, target_weights, bucket_id, True)
                    eval_ppx = math.exp(
                        eval_loss) if eval_loss < 300 else float('inf')
                    print(
                        "  eval: bucket %d perplexity %.2f" %
                        (bucket_id, eval_ppx))
                sys.stdout.flush()

                # save checkpoint
                self.save(global_step=epoch)

        # test loss
        for bucket_id in xrange(len(self._buckets)):
            encoder_inputs, target_weights = self.get_batch(
                self.test_set, bucket_id)
            _, eval_loss, _ = self.step(
                encoder_inputs, target_weights, bucket_id, True)
            eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            print("Test bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

    def get_batch(self, data, bucket_id):
        seq_length = self._buckets[bucket_id]
        encoder_size = seq_length  # , decoder_size = seq_length, seq_length + 1
        encoder_inputs = []
        for _ in xrange(self.batch_size):
            encoder_input = random.choice(data[bucket_id])
            encoder_input = encoder_input + [self.word2idx['_EOS']]
            encoder_pad_size = encoder_size - len(encoder_input)
            encoder_inputs.append(
                encoder_input + [self.word2idx['_PAD']] * encoder_pad_size)

        batch_encoder_inputs, batch_weights = [], []
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][
                length_idx] for batch_idx in xrange(self.batch_size)],
                dtype=np.int32))
            batch_weight = np.ones([self.batch_size], dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                if length_idx < encoder_size - 1:
                    target = encoder_inputs[batch_idx][length_idx + 1]
                if (length_idx == encoder_size - 1 or
                        target == self.word2idx['_PAD']):
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_weights

    def step(
            self,
            encoder_inputs,
            target_weights,
            bucket_id,
            forward_only=False):
        seq_length = self._buckets[bucket_id]
        encoder_size = seq_length  # , seq_length + 1

        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        # zero out last target
        input_feed[self.encoder_inputs[encoder_size].name] = np.zeros(
            [self.batch_size], dtype=np.float32)

        if not forward_only:
            output_feed = [
                self.updates[bucket_id],
                self.gradient_norms[bucket_id],
                self.losses[bucket_id]
            ]
        else:
            output_feed = [self.losses[bucket_id]]
            for l in xrange(encoder_size):
                output_feed.append(self.outputs[bucket_id][l])

        #outputs = session.run(output_feed, input_feed)
        outputs = self.sess.run(output_feed, input_feed)
        if not forward_only:
            # gradient norm, loss, (xent, -kl, annealing)
            return (outputs[1], outputs[2])
        else:
            return None, outputs[0], outputs[1:]  # gradient norm, loss, logits


#   def test(self):
        # print ('testing...')
        # self.create_model()
        # #self.batch_size = 1

        # # test loss
        # #TODO get for all
        # losses = []
        # for bucket_id in xrange(len(self._buckets)):
        # encoder_inputs, decoder_inputs, target_weights = self.get_batch(self.test_set, bucket_id)
        # _, eval_loss, _ = self.step(encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
        # eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
        # losses.append(eval_loss)
        # print("Test bucket %d perplexity %.2f" %(bucket_id, eval_ppx))
#     print("Test perplexity %.2f" %(math.exp(np.average(losses))))

    def test(self):
        print('testing...', self.forward_only)

        self.batch_size = 1
        self.create_model()
        self.dropout_prob = 1.0  # TODO same word_dropout as training??

        print(self.test_file)

        self.decode_single_sentence = True

        losses = []
        with open(self.test_file) as ftest:
            for idx, test_ids in enumerate(ftest):  # if idx > 10: break #TODO
                max_len_test_bucket = self._buckets[-1]
                test_ids = [int(x) for x in test_ids.strip().split()
                            ][:max_len_test_bucket - 1]
                inputs = [self.idx2word[input] for input in test_ids]
                print(' # %s ' % (' '.join(inputs)))
                bucket_id = min(
                    [b for b in xrange(len(self._buckets)) if self._buckets[b] > len(test_ids)])
                encoder_inputs, target_weights = self.get_batch(
                    {bucket_id: [test_ids]}, bucket_id)
                _, loss, logits = self.step(
                    encoder_inputstarget_weights, bucket_id, True)
                outputs = [int(np.argmax(logit, axis=1)) for logit in logits]

                if self.decode_single_sentence:
                    if self.word2idx['_EOS'] in outputs:
                        outputs = outputs[:outputs.index(self.word2idx['_EOS'])]
                outputs = [self.idx2word[output] for output in outputs]
                print(' > %s ' % (' '.join(outputs)))

                ppx = math.exp(loss) if loss < 300 else float('inf')
                print("%d %d ppl %.2f" % (idx, bucket_id, ppx))
                losses.append(loss)
        print("%.2f" % (math.exp(np.average(losses))))

    def sample(self):
        print('sampling...')
