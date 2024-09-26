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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy

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

def training_model(test_df, train_gen, valid_gen, test_gen, batch_size, steps_per_epoch, epochs):
    
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights="imagenet", pooling='avg')
    model = Sequential([
        base_model,
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(4, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    model.summary()
    
    train_gen = train_gen
    
    # Model CheckPoint
    checkpoint_cb = ModelCheckpoint('/trained_model/MyModel.keras', monitor='val_accuracy', save_best_only=True)

    # Early Stoping
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6)
    
    callbacks = [checkpoint_cb, earlystop_cb, reduce_lr]
    
    history = model.fit(train_gen, batch_size=batch_size, steps_per_epoch=steps_per_epoch, callbacks=callbacks,epochs=epochs, verbose=0, validation_data=valid_gen, validation_steps=15, shuffle=False)
    
    # test and evaluate
    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size
    
    train_score = model.evaluate(train_gen, steps=test_steps, verbose=1)
    valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=1)
    test_score = model.evaluate(test_gen, steps=test_steps, verbose=1)

    return model, train_score, valid_score, test_score


def save_model(model):

    subject = 'fruits_and_vegetables_classification'
    current_dir = os.path.dirname(__file__)
    save_path = os.path.join(current_dir, '..', 'trained_model')
    save_path = os.path.abspath(save_path)
    os.makedirs(save_path, exist_ok=True)

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