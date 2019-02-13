# !/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class Seq2SeqModel(object):
    def __init__(self, units, max_input_sequence_length, max_target_sequence_length, vocab_size, learning_rate,
                 start_token_idx, end_token_idx, embedding_size, beam_width=3, cell_type='lstm', optimizer='Adam'):
        """

        :param units: Number of hidden units in rnn cell. If a list is provided one layer with a cell for each entry
                      will be used.
        :param max_input_sequence_length: Maximum length of the input sequence.
        :param max_target_sequence_length: Maximum length of the target sequence.
        :param vocab_size: Size of the vocabulary.
        :param learning_rate: Learning rate.
        :param start_token_idx: Vocabulary index of the start token.
        :param end_token_idx: Vocabulary index of the end token.
        :param embedding_size: Size of the input embeddings.
        :param beam_width: Beam width.
        :param cell_type: RNN cell type. Possible values: 'rnn', 'gru' and 'lstm'.
        ;param optimizer: Optimizer.
        """
        self.units = units
        self.max_input_sequence_length = max_input_sequence_length
        self.max_target_sequence_length = max_target_sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.embedding_size = embedding_size
        self.beam_width = beam_width
        self.cell_type = cell_type
        self.opt = optimizer
        
        self.__build()
        
    def __build(self):
        """Build the graph for the model."""
        self.__declare_placeholders()
        self.__build_seq2seq()
        self.__compute_loss()
        self.__optimize()

    def __declare_placeholders(self):
        """Declare some necessary placeholders."""
        self.inputs = tf.placeholder(shape=(None, self.max_input_sequence_length), dtype=tf.int32, name='inputs')
        self.input_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='input_lengths')

        self.targets = tf.placeholder(shape=(None, self.max_target_sequence_length), dtype=tf.int32, name='targets')
        self.target_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='target_lengths')
        
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

        self.encoder_cell = tf.nn.rnn_cell.MultiRNNCell([Cell(num_units=units, name='encoder_cell_{}'.format(i))
                                                         for i, units in enumerate(self.units)])
        self.decoder_cell = tf.nn.rnn_cell.MultiRNNCell([Cell(num_units=units, name='decoder_cell_{}'.format(i))
                                                         for i, units in enumerate(self.units)])
        self.decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(cell=self.decoder_cell, output_size=self.vocab_size)

        self.embeddings = tf.get_variable(name='embedding_matrix', shape=[self.vocab_size, self.embedding_size],
                                          dtype=tf.float32, trainable=True)
        self.inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.inputs)

        _, self.final_encoder_states = tf.nn.dynamic_rnn(cell=self.encoder_cell, inputs=self.inputs_embedded,
                                                         sequence_length=self.input_lengths, dtype=tf.float32)

        start_tokens = tf.fill([tf.shape(self.inputs)[0]], self.start_token_idx)
        self.targets_as_inputs = tf.concat([tf.expand_dims(start_tokens, 1), self.targets], axis=1)
        self.targets_as_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.targets_as_inputs)

        self.training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.targets_as_inputs_embedded,
                                                                 sequence_length=self.target_lengths)
        self.training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell, helper=self.training_helper,
                                                                initial_state=self.final_encoder_states)

        self.train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=self.training_decoder,
                                                                     maximum_iterations=self.max_target_sequence_length,
                                                                     impute_finished=True)
        self.train_predictions = self.train_outputs.rnn_output
        pad_size = self.max_target_sequence_length - tf.shape(self.train_predictions)[1]
        self.train_predictions = tf.pad(self.train_predictions, [[0, 0], [0, pad_size], [0, 0]])

        final_encoder_states_tiled = tf.contrib.seq2seq.tile_batch(self.final_encoder_states,
                                                                   multiplier=self.beam_width)
        self.beam_search_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=self.decoder_cell,
                                                                        embedding=self.embeddings,
                                                                        start_tokens=start_tokens,
                                                                        end_token=self.end_token_idx,
                                                                        initial_state=final_encoder_states_tiled,
                                                                        beam_width=self.beam_width,
                                                                        length_penalty_weight=0)
        self.infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=self.beam_search_decoder,
                                                                     maximum_iterations=self.max_target_sequence_length,
                                                                     impute_finished=False)
        self.infer_predictions = tf.squeeze(self.infer_outputs.predicted_ids[:, :, 0])

    def __compute_loss(self):
        """Compute the loss."""
        mask = tf.cast(tf.sequence_mask(self.target_lengths, maxlen=self.max_target_sequence_length), dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.train_predictions, targets=self.targets, weights=mask)
    
    def __optimize(self):
        """Optimize the model."""
        if self.opt.lower() == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        
    def train_on_batch(self, session, input_batch, input_lengths, target_batch, target_lengths):
        """Train the model on the given batch.

        :param session: Tensorflow Session object.
        :param input_batch: Batch containing input sequences.
        :param input_lengths: Batch containing the length of the input sequences.
        :param target_batch: Batch containing target sequences.
        :param target_lengths: Array-like containing the length of the target sequences. Remaining characters are
                               ignored within loss computation.
        :return: Tuple containing the loss and the predictions.
        """
        feed_dict = {self.inputs: input_batch,
                     self.input_lengths: input_lengths,
                     self.targets: target_batch,
                     self.target_lengths: target_lengths}
        _, loss, predictions = session.run((self.train, self.loss, self.train_predictions), feed_dict=feed_dict)

        return loss, predictions

    def infer_on_batch(self, session, input_batch, input_lengths):
        """Infer predictions on batch.

        :param session: Tensorflow Session object.
        :param input_batch: Batch containing input sequences.
        :param input_lengths: Array-like containing the length of the input sequences.
        :return: Predictions.
        """
        feed_dict = {self.inputs: input_batch, self.input_lengths: input_lengths}
        predictions = session.run(self.infer_predictions, feed_dict=feed_dict)
        
        return predictions
