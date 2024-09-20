from fastapi import APIRouter
import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import time
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax

router = APIRouter()

# @router.get("/train")
# Define paths
def define_paths(data_dir):
    filepaths = []
    labels = []
    folders = os.listdir(data_dir)
    
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        filelist = os.listdir(folder_path)
        
        for file in filelist:
            fpath = os.path.join(folder_path, file)
            filepaths.append(fpath)
            labels.append(folder)
            
    return filepaths, labels


def define_df(files, classes):
    Fseries = pd.Series(files, name='filepaths')
    Lseries = pd.Series(classes, name='labels')
    return pd.concat([Fseries, Lseries], axis=1)


# Splitting data
def split_data(data_frame):
    stratify_df = data_frame['labels']
    train_df, remain_df = train_test_split(data_frame, train_size=0.8, shuffle=True, random_state=123, stratify=stratify_df)
    
    # valid and test dataframe
    stratify_df = remain_df['labels']
    valid_df, test_df = train_test_split(remain_df, test_size=0.5, shuffle=True, random_state=123, stratify=stratify_df)
    return train_df, valid_df, test_df

def create_generators(train_df, valid_df, test_df, batch_size):
    
    '''
        This function takes train, validation, and test dataframe and fit them into image data generator, 
        because model takes data from image data generator.Image data generator converts images into tensors. 
    '''
    
    # define model parameters
    img_size = (224, 224)
    channels = 3
    color = 'rgb'
    img_shape = (img_size[0], img_size[1], channels)
    
    # Recommended : use custom function for test data batch size, else we can use normal batch size.
    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))
    test_steps = ts_length // test_batch_size
    
    # This function which will be used in image data generator for data augmentation, it just take the image and return it again.
    def scalar(img):
        return img
    
    tr_gen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
    ts_gen = ImageDataGenerator(preprocessing_function=scalar)
    
    train_gen = tr_gen.flow_from_dataframe(
        train_df,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        class_mode='categorical',
        color_mode=color,
        shuffle=True,
        batch_size=batch_size
    )
    
    valid_gen = ts_gen.flow_from_dataframe(
        valid_df,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        class_mode='categorical',
        color_mode=color,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Note: we will use custom test_batch_size and make shuffle=False
    test_gen = ts_gen.flow_from_dataframe(
        test_df,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        class_mode='categorical',
        color_mode=color,
        shuffle=False,
        batch_size=test_batch_size
    )
    
    return train_gen, valid_gen, test_gen
    
class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, patience, stop_patience, threshold, factor, batches, epochs, ask_epoch):
        super(MyCallback, self).__init__()
        self._model = model  # Use a private attribute to store the model
        self.patience = patience
        self.stop_patience = stop_patience
        self.threshold = threshold
        self.factor = factor  # factor by which to reduce the learning rate
        self.batches = batches  # number of training batches to run per epoch
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        self.ask_epoch_initial = ask_epoch

        # callback variables
        self.count = 0  # how many times lr has been reduced without improvement
        self.stop_count = 0
        self.best_epoch = 1   # epoch with the lowest loss
        self.initial_lr = float(tf.keras.backend.get_value(self._model.optimizer.learning_rate))
        self.highest_tracc = 0.0
        self.lowest_vloss = np.inf
        self.best_weights = self._model.get_weights()
        self.initial_weights = self._model.get_weights()

    @property
    def model(self):
        return self._model

    def on_train_begin(self, logs=None):
        msg = 'Do you want model asks you to halt the training [y/n]?'
        print(msg)
        ans = input('')
        self.ask_permission = 1 if ans in ['Y', 'y'] else 0

        msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}'.format(
            'Epoch', 'Loss', 'Accuracy', 'V_loss', 'V_acc', 'LR', 'Next LR', 'Monitor', '% Improv', 'Duration')
        print(msg)
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        stop_time = time.time()
        tr_duration = stop_time - self.start_time
        hours = tr_duration // 3600
        minutes = (tr_duration - (hours * 3600)) // 60
        seconds = tr_duration - ((hours * 3600) + (minutes * 60))

        msg = f'Training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)'
        print(msg)

        # set the weights of the model to the best weights
        self.model.set_weights(self.best_weights)

    def on_train_batch_end(self, batch, logs=None):
        acc = logs.get('accuracy') * 100
        loss = logs.get('loss')

        msg = '{0:20s} Processing batch {1:} of {2:5s} - Accuracy: {3:5.3f} - Loss: {4:8.5f}'.format(
            ' ', str(batch), str(self.batches), acc, loss)
        print(msg, '\r', end='')

    def on_epoch_begin(self, epoch, logs=None):
        self.ep_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        ep_end = time.time()
        duration = ep_end - self.ep_start

        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        acc = logs.get('accuracy')
        v_acc = logs.get('val_accuracy')
        loss = logs.get('loss')
        v_loss = logs.get('val_loss')

        if acc < self.threshold:  # if training accuracy is below threshold adjust lr based on training accuracy
            monitor = 'accuracy'
            if epoch == 0:
                pimprov = 0.0
            else:
                pimprov = (acc - self.highest_tracc) * 100 / self.highest_tracc

            if acc > self.highest_tracc:  # training accuracy improved
                self.highest_tracc = acc
                self.best_weights = self.model.get_weights()
                self.count = 0
                self.stop_count = 0
                if v_loss < self.lowest_vloss:
                    self.lowest_vloss = v_loss
                self.best_epoch = epoch + 1

            else:
                if self.count >= self.patience - 1:  # adjust learning rate
                    lr = lr * self.factor
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr)
                    self.count = 0
                    self.stop_count += 1
                else:
                    self.count += 1

        else:  # training accuracy is above threshold
            monitor = 'val_loss'
            if epoch == 0:
                pimprov = 0.0
            else:
                pimprov = (self.lowest_vloss - v_loss) * 100 / self.lowest_vloss

            if v_loss < self.lowest_vloss:  # validation loss improved
                self.lowest_vloss = v_loss
                self.best_weights = self.model.get_weights()
                self.count = 0
                self.stop_count = 0
                self.best_epoch = epoch + 1
            else:
                if self.count >= self.patience - 1:
                    lr = lr * self.factor
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr)
                    self.stop_count += 1
                    self.count = 0
                else:
                    self.count += 1

                if acc > self.highest_tracc:
                    self.highest_tracc = acc

        msg = f'{str(epoch + 1):^3s}/{str(self.epochs):4s} {loss:^9.3f}{acc * 100:^9.3f}{v_loss:^9.5f}{v_acc * 100:^9.3f}{lr:^9.5f}{monitor:^11s}{pimprov:^10.2f}{duration:^8.2f}'
        print(msg)

        if self.stop_count > self.stop_patience - 1:
            msg = f'Training has been halted at epoch {epoch + 1} after {self.stop_patience} adjustments of learning rate with no improvement.'
            print(msg)
            self.model.stop_training = True

        if self.ask_epoch is not None and self.ask_permission != 0:
            if epoch + 1 >= self.ask_epoch:
                msg = 'Enter H to halt training or an integer for number of epochs to run then ask again:'
                print(msg)

                ans = input('')
                if ans in ['H', 'h']:
                    msg = f'Training has been halted at epoch {epoch + 1} due to user input.'
                    print(msg)
                    self.model.stop_training = True
                else:
                    try:
                        ans = int(ans)
                        self.ask_epoch += ans
                        msg = f'Training will continue until epoch {str(self.ask_epoch)}.'
                        print(msg)
                        print('{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}'.format(
                            'Epoch', 'Loss', 'Accuracy', 'V_loss', 'V_acc', 'LR', 'Next LR', 'Monitor', '% Improv', 'Duration'))
                    except Exception:
                        print('Invalid input.')

def training_model(test_df, train_gen, valid_gen, test_gen, batch_size, steps_per_epoch, epochs, patience, stop_patience, threshold, factor, ask_epoch):
    
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    
    base_model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(256, activation='relu'),
        Dropout(rate=0.45, seed=123),
        Dense(4, activation='softmax')
    ])

    model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    batches = int(np.ceil(len(train_gen.labels) / batch_size))
    
    callbacks = [MyCallback(model=model, patience=patience, stop_patience=stop_patience, threshold=threshold, factor=factor, batches=batches, epochs=epochs, ask_epoch=ask_epoch)]
    
    history = model.fit(train_gen, batch_size=batch_size, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=0, validation_data=valid_gen, validation_steps=None, shuffle=False)
    
    # test and evaluate
    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size
    
    train_score = model.evaluate(train_gen, steps=test_steps, verbose=1)
    valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=1)
    test_score = model.evaluate(test_gen, steps=test_steps, verbose=1)

    return model, history, train_score, valid_score, test_score


def save_model(model, test_score):

    subject = 'fruits_and_vegetables_classification'
    acc = test_score[1] * 100
    save_path = '/trained_model'

    # Save model
    save_id = str(f'{subject}.h5')
    model_save_loc = os.path.join(save_path, save_id)
    model.save(model_save_loc)
    print(f'model was saved as {model_save_loc}')

    # Save weights
    weight_save_id = str(f'{subject}.weights.h5')
    weights_save_loc = os.path.join(save_path, weight_save_id)
    model.save_weights(weights_save_loc)
    print(f'weights were saved as {weights_save_loc}')