import keras
from keras import layers
import tensorflow as tf
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.initializers import LecunNormal
from keras import backend as K


import numpy as np
import pandas as pd
import os
import random as python_random

SEED = 41
np.random.seed(SEED)
python_random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

def rmse(y_true, y_pred_):
    return K.sqrt(K.mean(K.square(y_pred_ - y_true), axis=1))

class MTL_model(Model):
    def __init__(self, signal_len, num_channels, lat_dim, trgt, index_list_input, hyperparameter_dict, Mode = 'MTL'):
        super(MTL_model, self).__init__()
        self.latent_dim = lat_dim
        self.signal_length = signal_len
        self.number_of_channels = num_channels
        self.target = trgt
        
        self.filter = hyperparameter_dict['filter_size']
        self.seed = SEED
        self.Mode = Mode
        self.kernel_size = hyperparameter_dict['kernel_size']
        self.encoder_af = hyperparameter_dict['encoder_af']
        self.decoder_af = hyperparameter_dict['decoder_af']
        self.classifier_af = hyperparameter_dict['classifier_af']
        self.index_list = index_list_input
        self.repetition = hyperparameter_dict['number_of_layers']
    

        
        ############################################################################
        # ------------------------------ Encoder --------------------------------

        inputs = layers.Input(shape=(self.signal_length, self.number_of_channels))

        input_list = []
        for index in self.index_list:
            a = index[0]
            b = index[1]
            x = layers.BatchNormalization(axis=-1)(inputs[:, :, a:b])

            for _ in range(self.repetition):
                #x = layers.Conv1D(self.filter, self.kernel_size, activation=self.encoder_af, 
                #                  kernel_initializer=LecunNormal(SEED), padding='same')(x)
                x = self.conv_layer()(x)
                x = layers.MaxPool1D(pool_size=5, strides=2, padding='same')(x)
                
            input_list.append(x)

        channel_y = layers.Concatenate(axis=-1)(input_list)

        lat_out = layers.Dense(self.number_of_channels, activation=self.encoder_af,
                               kernel_initializer=LecunNormal(self.seed),
                               kernel_regularizer=regularizers.l2(0.01))(channel_y)

        self.encoder = Model(inputs, lat_out, name='encoder')
        
        ############################################################################
        # ------------------------------ Classifier --------------------------------
        if self.Mode != 'AE':
            cls_input = layers.Input(shape=(self.latent_dim, self.number_of_channels))
            print(cls_input.shape)
            cls_o = layers.Flatten()(cls_input)
            cls_o = layers.BatchNormalization()(cls_o)
            cls_o = layers.Dense(self.target, kernel_regularizer=regularizers.l2(0.01))(cls_o)
            cls_output = layers.Softmax()(cls_o)
            print(cls_output.shape)
            self.classifier = Model(cls_input, cls_output, name='classifier')
            
        ############################################################################
        # ------------------------------ Decoder -------------------------------
        if self.Mode != 'classifier':
            latent_inputs = layers.Input(shape=(self.latent_dim, self.number_of_channels))
            print(latent_inputs.shape)
            output_list = []
            for index in self.index_list:
                a = index[0]
                b = index[1]
    
                x = layers.BatchNormalization(axis=-1)(latent_inputs[:, :, a:b])

                for _ in range(self.repetition):
                    #x = layers.Conv1DTranspose(self.filter, self.kernel_size, activation=self.decoder_af, 
                    #                   kernel_initializer=LecunNormal(SEED), strides=2, padding='same')(x)
                    x = self.deconv_layer()(x)
                output_list.append(x)
            channel_o = layers.Concatenate(axis=-1)(output_list)
    
            outputs = layers.Conv1D(self.number_of_channels, self.kernel_size, activation=self.decoder_af,
                                    kernel_initializer=LecunNormal(self.seed),
                                    kernel_regularizer=regularizers.l2(0.01), strides=1, padding='same')(channel_o)
        
    
            self.decoder = Model(latent_inputs, outputs, name='decoder')

    def conv_layer(self):

        if self.encoder_af == 'selu':
            lay = layers.Conv1D(self.filter, self.kernel_size, activation=self.encoder_af, strides=1, 
                          kernel_initializer=LecunNormal(self.seed),
                          padding='same')
        else:
            lay = layers.Conv1D(self.filter, self.kernel_size, activation=self.encoder_af, strides=1, padding='same')
    
        return lay

    def encoding_layer():
        enc_lay = tf.keras.Sequential()
        enc_lay.add(conv_layer())
        enc_lay.add(layers.MaxPool1D(pool_size=5, strides=2, padding='same'))

        return enc_layer

    def deconv_layer(self):
        if self.decoder_af == 'selu':
            lay = layers.Conv1DTranspose(self.filter, self.kernel_size, kernel_regularizer=regularizers.l2(0.01),
                                         kernel_initializer=LecunNormal(self.seed),
                                         activation=self.decoder_af, strides=2, padding='same')
        else:
            lay = layers.Conv1DTranspose(self.filter, self.kernel_size, kernel_regularizer=regularizers.l2(0.01),
                                         activation=self.decoder_af, strides=2, padding='same')
        return lay

    def __del__(self):
        print('Destructor called, AET deleted.')

    def call(self, x):
        print(self.Mode)
        if self.Mode == 'MTL':
            encoded = self.encoder(x)
            reconstructed = self.decoder(encoded)
            predicted = self.classifier(encoded)
            return reconstructed, predicted
        elif self.Mode == 'classifier':
            encoded = self.encoder(x)
            predicted = self.classifier(encoded)
            return predicted
        elif self.Mode == 'AE':
            encoded = self.encoder(x)
            reconstructed = self.decoder(encoded)
            return reconstructed

def mtl_model_training(dataset, hyperparameter_dict, weight_loss = None, Mode = 'MTL'):

    if hyperparameter_dict['multi_channel']:
        index_list = []
        for index in range(dataset['train']['data'].shape[2]):
            index_list.append([index, index + 1])
        print(index_list)
        
    else:
        index_list = [[0,dataset['train']['data'].shape[2]]]
        print(index_list)

    signal_length = dataset['train']['data'].shape[1]
    number_of_channels = dataset['train']['data'].shape[2]
    target = len(dataset['labels'])
    latent_dim = int(dataset['train']['data'].shape[1] / 2 ** hyperparameter_dict['number_of_layers'])
    optimizer = Adam(learning_rate=hyperparameter_dict['learning_rate'])

    if Mode == 'MTL':
        model = MTL_model(signal_length, number_of_channels, latent_dim, target, index_list, hyperparameter_dict, Mode = Mode)
        
        loss = {'output_1': 'Huber', 'output_2': tf.keras.losses.CategoricalCrossentropy(from_logits=False)}
        metrics= {'output_1': rmse, 'output_2': 'accuracy'}
        if weight_loss == None:
            loss_weights = None
        else:
            loss_weights = {'output_1': weight_loss[0], 'output_2': weight_loss[1]}
    
        to_train_x = dataset['train']['data']
        to_train_y = [dataset['train']['data'], dataset['train']['label']]
        to_test_x = dataset['validation']['data']
        to_test_y = [dataset['validation']['data'], dataset['validation']['label']]       
    
    elif Mode == 'classifier':
        model = MTL_model(signal_length, number_of_channels, latent_dim, target, index_list, hyperparameter_dict, Mode = Mode)
        
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        metrics= 'accuracy'
        
        to_train_x = dataset['train']['data']
        to_train_y = dataset['train']['label']
        to_test_x = dataset['validation']['data']
        to_test_y = dataset['validation']['label']
        loss_weights = None
    
    elif Mode == 'AE':
        model = MTL_model(signal_length, number_of_channels, latent_dim, target, index_list, hyperparameter_dict, Mode = Mode)
        
        loss = 'Huber'
        metrics= rmse
        
        to_train_x = dataset['train']['data']
        to_train_y = dataset['train']['data']
        to_test_x = dataset['validation']['data']
        to_test_y = dataset['validation']['data']
        loss_weights = None

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights = loss_weights)
    model.build((None, dataset['train']['data'].shape[1], dataset['train']['data'].shape[2]))
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001, restore_best_weights=True)
    history = model.fit(to_train_x, to_train_y, epochs=hyperparameter_dict['epochs'], batch_size=hyperparameter_dict['batch_size'],
                        validation_data=(to_test_x, to_test_y), shuffle=True, callbacks=[early_stopping])
    
    return model, history   