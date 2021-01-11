from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import shutil
import cv2
import os
import time
import random

# reference code: https://www.kaggle.com/alifrahman/covid-19-detection-using-transfer-learning
dataset_path = '../dataset'
covid_dataset_path = '../covid-chestxray-dataset'
normal_dataset_path = '../NIH_data'

csvPath = os.path.sep.join([covid_dataset_path, "metadata.csv"])
df_covid = pd.read_csv(csvPath)

# load metadata for images, the first one is the COVID dataset, the second on is from NIH dataset
csvPath = os.path.sep.join([covid_dataset_path, "metadata.csv"])
df_covid = pd.read_csv(csvPath)

csvPath = os.path.sep.join([normal_dataset_path, "Data_Entry_2017_v2020.csv"])
df_normal = pd.read_csv(csvPath)

# copy over all the COVID case images
# note that only images with "PA" view is used
for (i, row) in df_covid.iterrows():
    # if (1) the current case is not COVID-19 or (2) this is not
    # a 'PA' view, then ignore the row
    if (row["finding"].find("COVID-19")>=0) and row["view"] == "PA":
        

        # build the path to the input image file
        imagePath = os.path.sep.join([covid_dataset_path, "images", row["filename"]])

        # if the input image file does not exist (there are some errors in
        # the COVID-19 metadeta file), ignore the row
        if not os.path.exists(imagePath):
            continue

        # extract the filename from the image path and then construct the
        # path to the copied image file
        filename = row["filename"].split(os.path.sep)[-1]
        outputPath = os.path.sep.join([f"{dataset_path}/covid", filename])

        # copy the image
        shutil.copy2(imagePath, outputPath)
        
#load image from NIH data        
df_normal_pa = df_normal[df_normal['View Position']=='PA']
df_normal_pa['patientId'] = df_normal_pa['Image Index'].str.split('_').str[0]   
df_normal_pa_grouped = df_normal_pa.groupby('patientId').agg({'Image Index':'max'})
df_normal_pa_filetered = df_normal_pa[df_normal_pa['Image Index'].isin(df_normal_pa_grouped['Image Index'].unique().tolist())]

outputPath_Normal_list = []
for (i, row) in df_normal_pa_filetered.iterrows():
    # if (1) the current case is not COVID-19 or (2) this is not
    # a 'PA' view, then ignore the row
    if (row["Finding Labels"].find("No Finding")>=0) and row["View Position"] == "PA":        
        # build the path to the input image file
        imagePath = os.path.sep.join([normal_dataset_path, "images", row["Image Index"]])      
        if not os.path.exists(imagePath):
            continue
        # extract the filename from the image path and then construct the
        # path to the copied image file
        filename = row["Image Index"].split(os.path.sep)[-1]
        outputPath_Normal = os.path.sep.join([f"{dataset_path}/normal", filename])
        outputPath_Normal_list.append([imagePath, outputPath_Normal])


outputPath_Pneumonia_list = []
for (i, row) in df_normal.iterrows():

    if (row["Finding Labels"].find("Pneumonia")>=0) and row["View Position"] == "PA":
        

        # build the path to the input image file
        imagePath = os.path.sep.join([normal_dataset_path, "images", row["Image Index"]])
        if not os.path.exists(imagePath):
            continue
        # extract the filename from the image path and then construct the
        # path to the copied image file
        filename = row["Image Index"].split(os.path.sep)[-1]
        outputPath_Pneumonia = os.path.sep.join([f"{dataset_path}/pneumonia", filename])
        outputPath_Pneumonia_list.append([imagePath, outputPath_Pneumonia])

# the NIH dataset has large number of images, here we only sampled 500 for normal and pneumonia images respectively
outputPath_Normal_list_sample = random.sample(outputPath_Normal_list,500)
outputPath_Pneumonia_list_sample = random.sample(outputPath_Pneumonia_list,500)
# copy over the images
for pair in outputPath_Pneumonia_list_sample:
    shutil.copy2(pair[0], pair[1])

     
