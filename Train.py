# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:28:50 2019

@author: abousquet
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 20:01:46 2018

@author: abousquet
"""



import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import argparse
import random
import cv2
import os



def EntrainementReseau(CheminCSV,EPOCHS,BS,INIT_LR,WIDTH,HEIGHT):
    # initialize the data and labels
    #print("[INFO] téléchargement des CSV...")
    data = []
    labels = []
    
    random.seed(42)
    random.shuffle(CheminCSV)

    for CsvCour in CheminCSV:
        # MonCSV=csv.reader(open(CsvCour,"r") )
        
        
        MonCSV=np.genfromtxt(CsvCour, delimiter=';')
       
        MonCSV=np.delete(MonCSV,(0),axis=0) #supprimer la ligne d'identification
        MonCSV=np.delete(MonCSV,(0),axis=1)#suppression de la ligne timestamp 
        
        Imcour = cv2.resize(MonCSV, (WIDTH, HEIGHT))
        Imcour = img_to_array(Imcour)
        
        data.append(Imcour)
        
        label=CsvCour.split(os.path.sep)[-2]
        label = 1 if label == "leve_benne" else 0
        labels.append(label)
    
    
        
    # scale the raw pixel intensities to the range [0, 1]    
    data = np.array(data, dtype="float") / 2300
    labels = np.array(labels)
    
    
    
    
    
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
    	labels, test_size=0.25, random_state=42)
    
    # convert the labels from integers to vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)
    
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    	horizontal_flip=True, fill_mode="nearest")
    
    # initialize the model
    #print("[INFO] compiling model...")
    
    
    
    model = LeNet.build(width=3, height=600, depth=1, classes=2)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
    	metrics=["accuracy"])
    
    # train the network
    #print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    	epochs=EPOCHS, verbose=1)
    
    # save the model to disk
    #print("[INFO] serializing network...")
    model.save("ModeleCSV.model")
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="erreur à l'entrainement")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="erreur au test")
    plt.plot(np.arange(0, N), H.history["acc"], label="succés à l'entrainement")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="succés au test")
    plt.title("Précision du modèle")
    plt.xlabel("Nombre d'entrainement")
    plt.ylabel("Rapport perte/précision")
    plt.legend(loc="lower left")
    
    #EPOCHS = 50
    #INIT_LR = 1e-3
    #BS = 2
    
    CheminFichier=".\\Resultat_Apprentissage\\"+"Apprentissage_nbrsEntraiment_"+str(EPOCHS)+"_groupe_"+str(BS)+"_pourcentageAppInit_"+str(INIT_LR)+".png"
    plt.savefig(CheminFichier)
    plt.close()
