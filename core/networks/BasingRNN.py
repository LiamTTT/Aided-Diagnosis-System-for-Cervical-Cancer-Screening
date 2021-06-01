# -*- coding:utf-8 -*-
from keras.layers import Input, Dense 
from keras.layers.recurrent import SimpleRNN
from keras.models import Model


def RNN_classify(input_shape, ndims):
    input = Input(input_shape, name='feature_input')
    x = SimpleRNN(ndims, activation='relu', name='simple_rnn')(input)
    output = Dense(1, activation='sigmoid', name='output')(x)
    return Model(input, output)
