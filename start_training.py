# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

from src.model.chatbot import Chatbot
import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', '-e', type=int, metavar='epochs', required=True,
                        help='Number of epochs to train.')
    parser.add_argument('--name', '-n', type=str, metavar='name',
                        help='Name of the chatbot.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite if a chatbot with the given name already exists.')
    args = vars(parser.parse_args())

    if 'name' in args:
        name = args['name']
    else:
        name = config.chatbot_default_name

    if os.path.exists('model/'+name+'.ckpt.index'):
        assert args['overwrite'], "Chatbot '{}' already exists. Use option '--overwrite' to overwrite.".format(name)

    chatbot = Chatbot(name=name, vocabulary=config.vocabulary)
    chatbot.train(args['epochs'])

    print('Finished training.')
