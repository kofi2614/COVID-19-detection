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

# split data into training data testing set, convert target label to one-Hot encoding
def get_data(class_num):
    if class_num == 2:
        label_dict = {'covid':1,'normal':0,'pneumonia':0}
        label_list = ['normal','covid']
    elif class_num == 3:
        label_dict = {'covid':1,'normal':0,'pneumonia':2}
        label_list = ['normal','covid', 'pneumonia']
    label_transformed = np.vectorize(label_dict.get)(labels)
    (trainX, testX, trainY, testY) = train_test_split(data, label_transformed, test_size=0.2, stratify = label_transformed, random_state=42)
    trainY = to_categorical(trainY, num_classes = class_num)
    testY = to_categorical(testY, num_classes = class_num)
    trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")
    return trainX, trainY, testX, testY, trainAug, label_list
    
# bulid base model, each model has slightly different structure for the top layers.
def define_model(model_type, class_num):
    if model_type == 'ResNet50':
        baseModel = ResNet50(weights='imagenet',include_top=False, input_tensor=Input(shape=(224, 224, 3)), classes = class_num)
        headModel = baseModel.output
        headModel = GlobalAveragePooling2D()(headModel)
        headModel = Dense(1024, activation='relu')(headModel)
        predictions = Dense(class_num, activation='softmax')(headModel)
        model = Model(baseModel.input, predictions)

        return baseModel, model
        
    elif model_type == 'Vgg':
        baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)), classes = class_num)

        
        headModel = baseModel.output
        headModel = x= GlobalMaxPool2D()(headModel)
        headModel = Dense(512, activation="relu")(headModel)
        headModel= BatchNormalization()(headModel)
        headModel = Dropout(0.6)(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel= BatchNormalization()(headModel)
        headModel = Dropout(0.4)(headModel)
        headModel = Dense(64, activation="relu")(headModel)
        headModel= BatchNormalization()(headModel)
        headModel = Dropout(0.3)(headModel)
        headModel = Dense(class_num, activation="sigmoid")(headModel) #!!!
        
        model = Model(inputs=baseModel.input, outputs=headModel)
        
     

        return baseModel, model
    elif model_type == 'DenseNet':
        baseModel=DenseNet121(include_top=False, weights='imagenet',input_tensor=Input(shape=(224, 224, 3)),input_shape=None,pooling=None,classes = class_num)
        x=baseModel.output
        x= GlobalAveragePooling2D()(x)
        x= BatchNormalization()(x)
        x= Dropout(0.5)(x)
        x= Dense(1024,activation='relu')(x) 
        x= Dense(512,activation='relu')(x) 
        x= BatchNormalization()(x)
        x= Dropout(0.5)(x)

        preds=Dense(class_num,activation='softmax')(x) #FC-layer

        model=Model(inputs=baseModel.input,outputs=preds)
        return baseModel, model

        
    elif model_type == 'Inception':
        baseModel = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)),input_shape=None,pooling=None, classes = class_num)
        headModel = baseModel.output
        headModel = GlobalAveragePooling2D()(headModel)

        headModel = Dense(1024, activation='relu')(headModel)
        predictions = Dense(class_num, activation='softmax')(headModel)
        model = Model(baseModel.input, predictions)
        return baseModel, model
# plot the accuracy and loss for each epoch
def acc_loss_plot(H, model_type, class_num, addPhrase):
# plot the training loss and accuracy
    accs = H.history['accuracy']
    val_accs = H.history['val_accuracy']
    
    plt.plot(range(len(accs)),accs, label = 'Training_accuracy')
    plt.plot(range(len(accs)),val_accs, label = 'Validation_accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig("plot/{}_{}_class_{}_acc_plot.png".format(model_type, str(class_num), addPhrase))
    plt.close()
    
    loss = H.history['loss']
    val_loss = H.history['val_loss']

    plt.plot(range(len(accs)),loss, label = 'Training_loss')
    plt.plot(range(len(accs)),val_loss, label = 'Validation_loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig("plot/{}_{}_class_{}_loss_plot.png".format(model_type, str(class_num), addPhrase))
    plt.close()        

# train the model
def train_model(model, baseModel, model_type, trainX, trainY, testX, testY, trainAug, class_num):
    start_time = time.time()
    print("[INFO] compiling model...")
    
    # pre learning train
    EPOCHS = 30
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
    EPOCHS = 30
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
    
    return model, H


# plot the confusion matrix on test data with final trained model
def plot_confusion_matrix(model_type, class_num, cm, classes,
                      title='Normalized Confusion matrix',
                      cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, float("{:.2f}".format(cm[i, j])),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig('plot/{}_{}_class_Confusion_Matrix.png'.format(model_type, str(class_num)))
    plt.close()

# train model and predict on testing data    
def model_predict(model_type, class_num):
    trainX, trainY, testX, testY, trainAug, label_list = get_data(class_num)
    baseModel, model = define_model(model_type, class_num)
    model_final, H_final = train_model(model, baseModel, model_type, trainX, trainY, testX, testY, trainAug, class_num)
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
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(model_type, class_num, cnf_matrix, classes = label_list)


for class_num in [2, 3]:
    for model_type in ['Vgg','ResNet50','Inception','DensNet']:
        model_predict(model_type, class_num)    
    