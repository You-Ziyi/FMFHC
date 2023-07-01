import glob

import keras
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.initializers.initializers_v1 import TruncatedNormal
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model
from keras import backend as K

import os
import re

from keras.optimizer_v2.gradient_descent import SGD

class encoder_decoder(keras.Model):
    def __init__(self, encoder, decoder):
        super(encoder_decoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def return_encoder(self):
        return self.encoder
    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def load_data(path1,path2):
    files1 = glob.glob(path1 + '\\*.csv')
    files1.sort(key=natural_keys)

    files2 = glob.glob(path2 + '\\*.csv')
    files2.sort(key=natural_keys)

    x_train = pd.DataFrame()
    x_test = pd.DataFrame()

    i = 0
    l = []
    for file in files1:
        l.append(pd.read_csv(file, header=None))
    x_train = pd.concat(l)

    m = []
    for file in files2:
        m.append(pd.read_csv(file, header=None))
    x_test = pd.concat(m)

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    num1 = int(x_train.shape[0] / 64)
    num2 = int(x_test.shape[0] / 64)
    x_train = x_train.reshape(num1, 64, 64, 1)
    x_test = x_test.reshape(num2, 64, 64, 1)

    # x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    # x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())
    return x_train, x_test

def create_model():
    input_img = Input(shape=(64, 64, 1))
    # 编码器部分
    x = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2,2), padding='same')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x) # 16*16*16

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    encode_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(encoded)

    # 解码器部分
    input_decoder = Input(shape=(8, 8, 1))
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_decoder)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # 编码
    encoder_model = Model(input_img, encode_output)

    # 编码加解码

    decoder_model = Model(input_decoder, decoded)
    autoencoder = encoder_decoder(encoder_model, decoder_model)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return encoder_model, autoencoder

if __name__ == '__main__':


    # 实例化模型
    encoder_model, autoencoder = create_model()

    path1 = "*/xtrain"
    path2 = "*/xtest"

    x_train, x_test = load_data(path1, path2)

    from keras.callbacks import TensorBoard
    autoencoder.fit(x_train, x_train,
                    epochs=30,  # 自己定
                    batch_size=128,  # 自己定
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    # 重建图片
    import matplotlib.pyplot as plt

    enc_model = autoencoder.return_encoder()
    # 编码
    encoded_imgs = enc_model.predict(x_test)
    # 解码
    decoded_imgs = autoencoder.predict(x_test)

    enc_model.save("*/encoder.h5")


    #autoencoder.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    #autoencoder.compile(optimizer='adam', loss='binary_crossentropy')