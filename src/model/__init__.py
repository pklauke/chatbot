# !/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

# Fix bugs in newer tensorflow versions
setattr(tf.contrib.rnn.RNNCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)
setattr(tf.nn.rnn_cell.RNNCell, '__deepcopy__', lambda self, _: self)
setattr(tf.nn.rnn_cell.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.nn.rnn_cell.LSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.nn.rnn_cell.MultiRNNCell, '__deepcopy__', lambda self, _: self)
