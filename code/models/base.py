import os
import sys
import argparse
import path
import numpy as np
from collections import defaultdict, OrderedDict
import tensorflow as tf


class Base():
    def __init__(self):
        self.reload = False
        #self.summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)

    def init_summary(self):

        model_dir, attr_dir = self.get_model_dir()
        log_dir = os.path.join(self.log_dir, model_dir, attr_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.summary_writer = tf.summary.FileWriter(
            log_dir + '/' + self.mode, graph=tf.get_default_graph())

    def get_model_dir(self):
        data_model_dir = self.dataset + '/' + self.name
        attr_dirs = []
        for attr in self._attrs:
            if hasattr(self, attr):
                attr_dirs.append("%s=%s" % (attr, getattr(self, attr)))
        attr_dir = '/'.join(attr_dirs)
        return data_model_dir, attr_dir

    def save(self, sess, global_step=None):
        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__
        model_dir, attr_dir = self.get_model_dir()
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir, attr_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            sess,
            os.path.join(
                checkpoint_dir,
                model_name),
            global_step=global_step)

    def get_parameter_size(self):
        total_parameters = 0
        variables = []
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variables.append(variable.name)
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return variables, total_parameters

    def initialize(self, sess):
        print('Intializing parameters...')
        model_name = type(self).__name__
        model_dir, attr_dir = self.get_model_dir()
        log_dir = os.path.join(self.log_dir, model_dir, attr_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)


        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        self.load(sess, self.checkpoint_dir)

    def load(self, sess, checkpoint_dir, checkpoint_step=None):
        model_dir, attr_dir = self.get_model_dir()
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir, attr_dir)
        print(" [*] Loading checkpoints:", checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path and self.reload:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if checkpoint_step:
                for ckpt_n in ckpt.all_model_checkpoint_paths:
                    if int(ckpt_n.split('-')[-1]) == checkpoint_step:
                        ckpt_name = os.path.basename(ckpt_n)
                        print(' [!] Found!!', ckpt_n, self.global_step.eval())
                        break
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Load SUCCESS: " + ckpt_name)
            return True
        else:
            print(" [!] Load FAIL...")
            return False

    def add_train_op(self, lr_method, lr, loss, clip=-1):
        _lr_m = lr_method.lower()  # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam':  # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr, 1e-5)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                train_op = optimizer.minimize(loss)
        return train_op
