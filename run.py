# !/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify

from src.model.chatbot import Chatbot
import config

app = Flask(__name__)

chatbot = Chatbot(name=config.chatbot_default_name, vocabulary=config.vocabulary)
chatbot.load()


@app.route("/", methods=['POST'])
def request_answer():
    try:
        message = request.json['question']
    except KeyError:
        return "Given JSON has wrong format. Please provide a JSON with key 'question'. E.g. " \
               "'{'question': 'how are you'}'"
    reply = chatbot.get_reply(message)

    return jsonify({'answer': reply})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=False)
