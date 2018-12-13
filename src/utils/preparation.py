# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains classes that serve the chatbot for preparing.

Currently implemented classes:

CharacterLevelPreparer
"""

from abc import ABC, abstractmethod
from typing import Union, Dict, List

import numpy as np

from src.utils.chatbot_utils import prepare_sentence, map_char_sentences_to_index, map_indices_to_char_sentence
import config


class BasePreparer(ABC):
    def __init__(self, sample_to_index: Dict, index_to_sample: Union[List, Dict]):
        self.sample_to_index = sample_to_index
        self.index_to_sample = index_to_sample

    @abstractmethod
    def prepare_messages(self, messages: Union[List[str], np.ndarray]):
        pass

    @abstractmethod
    def prepare_replies(self, replies: Union[List[List[int]], np.ndarray]):
        pass


class CharacterLevelPreparer(BasePreparer):
    def __init__(self, char_to_index: Dict[str, int], index_to_char: Union[List, Dict[int, str]]):
        super().__init__(char_to_index, index_to_char)

    def prepare_messages(self, messages: Union[List[str], np.ndarray]):
        """Prepare the message for the chatbot.

        :param messages: Array-like containing strings.
        :returns: 2D np.ndarray containing the prepared message.
        """
        super().prepare_messages(messages)
        messages = [[char.lower() for char in message if char.lower() in self.sample_to_index] for message in messages]
        messages = [prepare_sentence(message, **config.tokens, max_len=config.max_sequence_length)
                    for message in messages]
        messages = np.array(map_char_sentences_to_index(messages, self.sample_to_index))

        return messages

    def prepare_replies(self, replies: Union[List[List[int]], np.ndarray]):
        """Prepare the reply of the chatbot.

        :param replies: 2D array-like with shape () containing the predicted characters.
        :returns: String containing the prepared reply.
        """
        super().prepare_replies(replies)
        replies = map_indices_to_char_sentence(replies, self.index_to_sample)
        for i, reply in enumerate(replies):
            cleaned_reply = ''
            for char in reply:
                if char == config.end_token:
                    break
                else:
                    cleaned_reply += char
                    cleaned_reply = cleaned_reply.replace(config.start_token, '').replace(config.pad_token, '')
            replies[i] = cleaned_reply
        return replies
