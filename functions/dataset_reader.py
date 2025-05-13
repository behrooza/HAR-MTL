import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import os
import random as python_random
from keras import utils

SEED = 41
np.random.seed(SEED)
python_random.seed(SEED)
utils.set_random_seed(SEED)


def get_uchar_data():
    path = 'D:/data recording/UCI HAR Dataset'
    train_dir = '%s/UCI HAR Dataset/train/Inertial Signals'%path
    test_dir = '%s/UCI HAR Dataset/test/Inertial Signals'%path
    columns = ['body_acc_x', 'body_acc_y', 'body_acc_z',
               'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
               'total_acc_x', 'total_acc_y', 'total_acc_z'
               ]
    y_train = pd.read_csv('%s/UCI HAR Dataset/train/y_train.txt'%path, sep='\s+', header=None)
    y_train = np.array(y_train).reshape((y_train.shape[0],))
    train = np.empty((y_train.shape[0], 128, len(columns)))
    y_test = pd.read_csv('%s/UCI HAR Dataset/test/y_test.txt'%path, sep='\s+', header=None)
    y_test = np.array(y_test).reshape((y_test.shape[0],))
    test = np.empty((y_test.shape[0], 128, len(columns)))
    i = 0
    for col in columns:
        X_tr = pd.read_csv('%s/%s_train.txt' % (train_dir, col), sep='\s+', header=None)
        train[:, :, i] = np.array(X_tr)

        X_ts = pd.read_csv('%s/%s_test.txt' % (test_dir, col), sep='\s+', header=None)
        test[:, :, i] = np.array(X_ts)
        i += 1

    y_train -= 1  # bring the labels in the range of 0 to 5
    y_test -= 1

    total_acc = np.sqrt(train[:, :, -3] ** 2 + train[:, :, -2] ** 2 + train[:, :, -1] ** 2)
    train = np.concatenate((train, np.expand_dims(total_acc, -1)), axis=-1)

    total_acc = np.sqrt(test[:, :, -1] ** 2 + test[:, :, -2] ** 2 + test[:, :, -3] ** 2)
    test = np.concatenate((test, np.expand_dims(total_acc, -1)), axis=-1)
    columns += ['total_acc']
   
    for i in range(train.shape[-1]):
        scaler = preprocessing.StandardScaler()
        train[:, :, i] = scaler.fit_transform(train[:, :, i])
        test[:, :, i] = scaler.transform(test[:, :, i])
    
    activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']

    y_train = utils.to_categorical(y_train, len(activity_labels))
    y_test = utils.to_categorical(y_test, len(activity_labels))

    train_data, val_data, train_label, val_label = train_test_split(train, y_train, test_size=0.2,
                                                                              random_state=SEED,
                                                                              shuffle=True)

    train_dict = {'data': train_data, 'label': train_label}

    val_dict = {'data': val_data, 'label': val_label}

    test_dict = {'data': test, 'label': y_test}
    
    dataset = {'train': train_dict,
              'validation': val_dict,
              'test': test_dict,
              'labels': activity_labels}    

    return dataset
    