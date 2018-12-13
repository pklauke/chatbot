# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from src.model.chatbot import Chatbot
import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', '-e', type=int, metavar='epochs', required=True,
                        help='Number of epochs to train.')
    parser.add_argument('--name', '-n', type=str, metavar='name',
                        help='Name of the chatbot.')
    args = vars(parser.parse_args())

    if 'name' in args:
        name = args['name']
    else:
        name = config.chatbot_default_name

    chatbot = Chatbot(name=name, vocabulary=config.vocabulary)
    chatbot.load()
    chatbot.train(args['epochs'])

    print('Finished training.')
