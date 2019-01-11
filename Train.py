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




# initialize the number of epochs to train for, initia learning rate,
# and batch size

EPOCHS = 50
INIT_LR = 1e-3
BS = 2
WIDTH=3
HEIGHT=600




# initialize the data and labels
print("[INFO] téléchargement des CSV...")
data = []
labels = []


CheminCSV=sorted(list(paths.list_files(".\Data",".csv") ) )
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
    """
    data = pd.read_csv(CsvCour).values
    """
    label=CsvCour.split(os.path.sep)[-2]
    label = 1 if label == "leve_benne" else 0
    labels.append(label)


"""
toto=data.shape[0]
data = data[:, 1:].reshape(data.shape[0],1,28, 28).astype( 'float32' )
"""
    
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
print("[INFO] compiling model...")



model = LeNet.build(width=3, height=600, depth=1, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save("ModeleCSV.model")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("VisuApprentissage.png")

