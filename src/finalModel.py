import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19, DenseNet121, DenseNet201, ResNet50
from tensorflow.keras.layers import Activation, Dropout,Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Lambda, Input, AveragePooling2D, GlobalAveragePooling2D,GlobalMaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import shutil
import cv2
import os
import time
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import itertools
import modelTrain
# reference code: https://www.kaggle.com/alifrahman/covid-19-detection-using-transfer-learning
# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 30
BS = 3

#load data
dataset_path = '../data/spark_output'

data = []
labels = []
print("[INFO] loading images...")
for xtype in ['covid','normal','pneumonia']:
    print('parsing dataset ' + xtype)
    # read in flatten array
    df_flatarr = pd.read_csv(dataset_path+'/'+xtype+'/data.csv.gz', header=None)
    # reshape
    arr = np.reshape(df_flatarr.values, (len(df_flatarr),224,224,3))
    # append
    data.append(arr)
    labels += [xtype]*len(df_flatarr)
data = np.vstack(data)
labels = np.array(labels)


def train_model(model, baseModel, model_type, trainX, trainY, testX, testY, trainAug, class_num):
    start_time = time.time()
    print("[INFO] compiling model...")
    
    # pre learning train
    EPOCHS = 10
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    baseModel.trainable = False
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    print("[INFO] Pretraining...")
    H = model.fit_generator(
        trainAug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=trainAug.flow(testX, testY, batch_size=BS),
#         callbacks=[anne, checkpoint],
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)
    acc_loss_plot(H, model_type, class_num,'pretrained')
    
    # transfer learning
    EPOCHS = 100
    baseModel.trainable = True
    opt = Adam(lr=INIT_LR/10, decay=INIT_LR / (EPOCHS * 100))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    print("[INFO] Fine turinng...")
    H = model.fit_generator(
        trainAug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=trainAug.flow(testX, testY, batch_size=BS),
#         callbacks=[anne, checkpoint],
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)
    acc_loss_plot(H, model_type, class_num, 'fine-tuning')
    
    # smaller learning rate
    
    EPOCHS = 100
    baseModel.trainable = True
    opt = Adam(lr=INIT_LR/100, decay=INIT_LR / (EPOCHS * 100))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    print("[INFO] Second Fine turinng...")
    H = model.fit_generator(
        trainAug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=trainAug.flow(testX, testY, batch_size=BS),
#         callbacks=[anne, checkpoint],
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    acc_loss_plot(H, model_type, class_num, 'second fine-tuning')
#     retrain model with optimized epoch
    return model, H



def model_predict(model_type, class_num):
    trainX, trainY, testX, testY, trainAug, label_list = modelTrain.get_data(class_num)
    baseModel, model = modelTrain.define_model(model_type, class_num)
    model_final, H_final = modelTrain.train_model(model, baseModel, model_type, trainX, trainY, testX, testY, trainAug, class_num)
    model_final.save(model_type)
    predictY = model_final.predict(testX)
    y_true = []
    for y in testY:
        y_true.append(list(y).index(max(y)))
    y_pred = []
    for y in predictY:
        y_pred.append(list(y).index(max(y)))
    report = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv('report/{}_{}_class_report.csv'.format(model_type, str(class_num)))
    cnf_matrix = modelTrain.confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(model_type, class_num, cnf_matrix, classes = label_list)

model_predict('DenseNet',3)
    