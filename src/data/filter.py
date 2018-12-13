# !/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np

import config


class DataFilter(ABC):
    """Abstract class for filtering data."""
    @abstractmethod
    def filter_data(self, data):
        pass


class CharacterLevelFilter(DataFilter):
    """Class to filter data for a character-level sequence to sequence model."""
    def filter_data(self, data):
        """Filter the given data.

        :param data: Data to filter with shape (nrows, 2). The first column is assumed to be input data and the second
                     target data.
        :return: Tuple containing the filtered input and target data.
        """

        input_words = [sentence.split() for sentence in data[:, 0]]
        target_words = [sentence.split() for sentence in data[:, 1]]

        words, counts = np.unique([word for sentence in input_words + target_words for word in sentence],
                                  return_counts=True)
        word_counts = {word: count for word, count in zip(words, counts)}

        new_input, new_target = [], []
        for i, (input_sentence, target_sentence) in enumerate(zip(input_words, target_words)):
            if (len(input_sentence) >= config.filter_min_words
                    and len(target_sentence) >= config.filter_min_words
                    and len(' '.join(input_sentence)) >= config.filter_min_sequence_length
                    and len(' '.join(target_sentence)) >= config.filter_min_sequence_length
                    and len(' '.join(input_sentence)) <= config.filter_max_sequence_length
                    and len(' '.join(target_sentence)) <= config.filter_max_sequence_length
                    and all([word_counts[word] >= config.filter_min_count_per_word for word in input_sentence])
                    and all([word_counts[word] >= config.filter_min_count_per_word for word in target_sentence])):
                new_input.append(input_sentence)
                new_target.append(target_sentence)
        print('{}/{} rows kept'.format(len(new_input), len(input_words)))

        input_chars = [' '.join(sentence) for sentence in new_input]
        target_chars = [' '.join(sentence) for sentence in new_target]

        return input_chars, target_chars
