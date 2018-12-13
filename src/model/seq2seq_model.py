# !/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class Seq2SeqModel(object):
    def __init__(self, units, max_input_sequence_length, max_target_sequence_length, vocab_size, learning_rate,
                 embedding_size=100, cell_type='lstm', optimizer='Adam'):
        """

        :param units: Number of hidden units in rnn cell. If a list is provided one layer with a cell for each entry
                      will be used.
        :param max_input_sequence_length: Maximum length of the input sequence.
        :param max_target_sequence_length: Maximum length of the target sequence.
        :param vocab_size: Size of the vocabulary.
        :param learning_rate: Learning rate.
        :param embedding_size: Embedding size to use for the input.
        :param cell_type: RNN cell type. Possible values: 'rnn', 'gru' and 'lstm'.
        ;param optimizer: Optimizer.
        """
        self.units = units
        self.max_input_sequence_length = max_input_sequence_length
        self.max_target_sequence_length = max_target_sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.cell_type = cell_type
        self.optimizer = optimizer
        
        self.__build()
        
    def __build(self):
        """Build the graph for the model."""
        self.__declare_placeholders()
        self.__build_seq2seq()
        self.__compute_loss()
        self.__optimize()
        
    def __declare_placeholders(self):
        """Declare some necessary placeholders."""
        self.target_batch = tf.placeholder(shape=(None, None), dtype=tf.int32, name='target_batch')
        self.target_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='target_sentence_length')
        
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder_inputs_{}'.format(i)) 
                               for i in range(self.max_input_sequence_length)]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder_inputs_{}'.format(i)) 
                               for i in range(self.max_target_sequence_length)]
        
    def __build_seq2seq(self):
        """Build the graph for the sequence to sequence model.

        Creates an embedding for the input.

        :return: None
        """
        if self.cell_type.lower() == 'rnn':
            Cell = tf.nn.rnn_cell.RNNCell
        elif self.cell_type.lower() == 'gru':
            Cell = tf.contrib.rnn.GRUCell
        elif self.cell_type.lower() == 'lstm':
            Cell = tf.nn.rnn_cell.LSTMCell
        self.rnn = tf.nn.rnn_cell.MultiRNNCell([Cell(num_units=units) for units in self.units])

        def __build_encoder_decoder(encoder_inputs, decoder_inputs, reuse=None, feed_previous=False):

            with tf.variable_scope("embedding_rnn_seq2seq", reuse=reuse):
                rnn_outputs, _ = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                    encoder_inputs, decoder_inputs, self.rnn,
                    num_encoder_symbols=self.vocab_size, num_decoder_symbols=self.vocab_size,
                    embedding_size=self.embedding_size, feed_previous=feed_previous, dtype=tf.float32
                )
            outputs = [tf.reshape(output, shape=(-1, 1, self.vocab_size)) for output in rnn_outputs]
            outputs = tf.concat(outputs, axis=1)

            return outputs

        self.train_outputs = __build_encoder_decoder(self.encoder_inputs, self.decoder_inputs)
        self.train_probabilities = tf.nn.softmax(self.train_outputs)

        self.infer_outputs = __build_encoder_decoder(self.encoder_inputs, self.encoder_inputs,
                                                     reuse=True, feed_previous=True)
        self.infer_probabilities = tf.nn.softmax(self.infer_outputs)

    def __compute_loss(self):
        """Compute the loss."""
        mask = tf.sequence_mask(self.target_length, maxlen=self.max_target_sequence_length)
        self.loss = tf.contrib.seq2seq.sequence_loss(self.train_outputs, self.target_batch,
                                                     tf.cast(mask, dtype=tf.float32))
    
    def __optimize(self):
        """Optimize the model."""
        self.train = tf.contrib.layers.optimize_loss(self.loss, tf.train.get_global_step(), self.learning_rate, 
                                                     optimizer=self.optimizer, clip_gradients=1.0)
        
    def train_on_batch(self, session, input_batch, target_batch, target_length, return_probabilities=False):
        """Train the model on the given batch.

        :param session: Tensorflow Session object.
        :param input_batch: Batch containing input sequences.
        :param target_batch: Batch containing target sequences.
        :param target_length: Batch containing the length of the target sequences. Remaining characters are ignored
                              within loss computation.
        :param return_probabilities: Boolean indicating if probabilities want to be returned. If `False softmax
                                     computation is skipped and raw logits are returned.
        :return: Tuple containing the loss and the prediction probabilities
        """
        feed_dict = {**{key: value for (key, value) in zip(self.encoder_inputs, input_batch)},
                     **{key: value for (key, value) in zip(self.decoder_inputs, target_batch[:-1, :])},
                     self.target_batch: target_batch[1:, :].T,
                     self.target_length: target_length}
        if return_probabilities:
            _, loss, predictions = session.run((self.train, self.loss, self.train_probabilities), feed_dict=feed_dict)
        else:
            _, loss, predictions = session.run((self.train, self.loss, self.train_outputs), feed_dict=feed_dict)
        return loss, predictions

    def infer_on_batch(self, session, input_batch):
        """Infer predictions on batch.

        :param session: Tensorflow Session object.
        :param input_batch: Batch containing input sequences.
        :return: Prediction probabilities.
        """
        feed_dict = {**{key: value for (key, value) in zip(self.encoder_inputs, input_batch)}}
        prediction_proba = session.run(self.infer_probabilities, feed_dict=feed_dict)
        
        return prediction_proba
