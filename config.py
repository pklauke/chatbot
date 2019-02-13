# !/usr/bin/env python
# -*- coding: utf-8 -*-

cell_type = 'lstm'  # possible values: 'lstm', 'gru', 'rnn'
hidden_units = [64, 64]
beam_width = 3

dataset_filename = '__data__/data.feather'

start_token = '$'
end_token = '#'
pad_token = '^'
tokens = dict(start_token=start_token, end_token=end_token, pad_token=pad_token)
vocabulary = " abcdefghijklmnopqrstuvwxyz1234567890$#^,?;:!'\"/()%<>+-*"

# data filter criteria
filter_min_count_per_word = 100
filter_min_sequence_length = 15
filter_max_sequence_length = 50
filter_min_words = 3

max_sequence_length = 50
embedding_size = len(vocabulary) - len(tokens)

learning_rate = 0.001
optimizer = 'adam'
batch_size = 128

chatbot_default_name = 'chatbot'

test_messages = ['What is your name?', 'How are you?', 'What are your hobbies?']
