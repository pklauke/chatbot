# !/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from src.model.seq2seq_model import Seq2SeqModel
import config
from src.utils.preparation import CharacterLevelPreparer
from src.data.filter import CharacterLevelFilter


class Chatbot(object):
    """Chatbot class using sequence to sequence model for replying to messages.

    :param name: Name of the Chatbot. Identifier to continue training later on.
    :param vocabulary: Vocabulary of the Chatbot."""
    def __init__(self, name, vocabulary):
        self.name = name
        self.saver = None

        self.index_to_sample = [sample for sample in vocabulary]
        self.sample_to_index = {char: index for index, char in enumerate(self.index_to_sample)}
        self.preparer = CharacterLevelPreparer(char_to_index=self.sample_to_index, index_to_char=self.index_to_sample)

        self.filter = CharacterLevelFilter()

        self.model = Seq2SeqModel(config.hidden_units, config.max_sequence_length + 1, config.max_sequence_length + 1,
                                  learning_rate=config.learning_rate,
                                  start_token_idx=config.vocabulary.index(config.start_token),
                                  end_token_idx=config.vocabulary.index(config.end_token),
                                  vocab_size=len(vocabulary), embedding_size=config.embedding_size,
                                  beam_width=config.beam_width, cell_type=config.cell_type, optimizer=config.optimizer)
        self.session = tf.Session()

    def save(self):
        """Save the Chatbot."""
        self.saver = tf.train.Saver()
        self.saver.save(self.session, 'model/'+self.name+'.ckpt')

    def load(self):
        """Load the Chatbot."""
        self.saver = tf.train.Saver()
        self.saver.restore(self.session, 'model/'+self.name+'.ckpt')

    def train(self, epochs):
        """Train the Chatbot using the parameters in the config. The Chatbot is automatically saved after each epoch.

        :param epochs: Number of epochs to train.
        """
        data = pd.read_feather(config.dataset_filename).values
        input_sequences, target_sequences = self.filter.filter_data(data)
        del data

        input_sequences_length = np.array([min(len(sentence), config.max_sequence_length)
                                           for sentence in input_sequences])
        input_sequences = self.preparer.prepare_messages(input_sequences)
        target_sequences_length = np.array([min(len(sentence), config.max_sequence_length)
                                            for sentence in target_sequences])
        target_sequences = self.preparer.prepare_messages(target_sequences)

        init = tf.global_variables_initializer()

        self.session.run(init)

        for epoch in range(epochs):
            t0 = datetime.datetime.now()

            n_batches = input_sequences.shape[0] // config.batch_size + 1

            for batch_index in range(n_batches):

                batch_begin = batch_index * config.batch_size
                batch_end = min((batch_index + 1) * config.batch_size, input_sequences.shape[0])

                if batch_begin < batch_end:
                    prep_input = input_sequences[batch_begin:batch_end, :]
                    prep_target = target_sequences[batch_begin:batch_end, :]

                    loss, prediction_train = self.model.train_on_batch(
                        self.session,
                        input_batch=prep_input,
                        input_lengths=input_sequences_length[batch_begin:batch_end] + 1,
                        target_batch=prep_target,
                        target_lengths=target_sequences_length[batch_begin:batch_end] + 1
                    )

            self.save()

            t1 = datetime.datetime.now()
            elapsed_time = (t1 - t0).seconds / 60
            print('Epoch: {}; loss: {:0.3f}; time[min]: {:0.1f}'.format(epoch, loss, elapsed_time))

            if config.test_messages:
                test_messages_lengths = np.array([min(len(sentence), config.max_sequence_length)
                                                  for sentence in config.test_messages])
                test_messages = np.array(self.preparer.prepare_messages(config.test_messages))
                indices = self.model.infer_on_batch(self.session, input_batch=test_messages,
                                                    input_lengths=test_messages_lengths)
                replies = self.preparer.prepare_replies(indices)

                for message, reply in zip(config.test_messages, replies):
                    print('\tMessage:', message)
                    print('\tReply:  ', reply, '\n')

    def get_reply(self, message):
        """Get a reply from the Chatbot for a message.

        :param message: Message to the Chatbot.
        :returns: Reply to the message.
        """
        input_lengths = np.array([min(len(message), config.max_sequence_length)])
        input_batch = np.array(self.preparer.prepare_messages([message]))

        predictions = self.model.infer_on_batch(self.session, input_batch, input_lengths)

        reply = self.preparer.prepare_replies(predictions)

        return reply
