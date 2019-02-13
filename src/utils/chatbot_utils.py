# !/usr/bin/env python
# -*- coding: utf-8 -*-


def prepare_sentence(sentence, end_token='#', pad_token='^', max_len=40):
    """Prepares a sentence. Clips it to the given maximum length and adds start, end and padding tokens.

    Prepared sentence will have length `max_len`+1.

    :param sentence: Sentence to prepare.
    :param start_token: Start token to start each sentence with.
    :param end_token: End token to end each sentence with.
    :param pad_token: Padding tokens that are added to the sentence if the sentence is shorter than `max_len`.
    :param max_len: Maximum Sentence length. Longer sentences are clipped. Padding tokens are added if the sentence
                    is shorter. Start and end tokens aren't added to the sentence length.
    """
    prepared_sentence = []
    prepared_sentence += sentence[:max_len]
    prepared_sentence += [end_token]
    prepared_sentence += [pad_token] * (max_len+1-len(prepared_sentence))

    return prepared_sentence


def map_char_sentences_to_index(sentences, char_to_index):
    """Map sentences split by individual characters using a dictionary.

    :param sentences: List containing lists with character sentences.
    :param char_to_index: Dictionary to use for mapping.
    """
    return [[char_to_index[char.lower()] for char in sentence]
            for sentence in sentences]


def map_indices_to_char_sentence(indices, index_to_char):
    """Map indices to sentences split by individual characters.

    :param indices: List containing lists with indices.
    :param index_to_char: Dictionary or list to use for mapping.
    """
    return [[index_to_char[index] for index in sentence_indices] for sentence_indices in indices]
