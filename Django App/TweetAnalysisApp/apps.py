from django.apps import AppConfig
from django.conf import settings
import os
import pickle
import tensorflow as tf


class TweetanalysisappConfig(AppConfig):
    name = 'TweetAnalysisApp'
    path = os.path.join(settings.TOKEN, 'token.p')

    with open(path, 'rb') as pickled:
        data = pickle.load(pickled)
    tokenizer = data['tokenizer']
    pad_sequences = data['pad_sequences']

    my_model = tf.keras.models.load_model(settings.MODELS)
