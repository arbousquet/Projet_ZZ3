# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 19:41:26 2018

@author: abousquet
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from imutils import paths
"""
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

"""



# load the image

#image = cv2.imread("D:\\hackatonSopra2018\\testML\\image-classification-keras\\imagesMais\\MaisNonPret\\00000001.jpg")#args["image"]
#orig = image.copy()

# pre-process the image for classification


# load the trained convolutional neural network
def TestReseau(CheminCSV,EPOCHS,BS,INIT_LR,WIDTH, HEIGHT):
    #print("[INFO] loading network...")
    model = load_model(".\ModeleCSV.model")#args["model"]
    
    CheminFichier=".\\Resultat_Apprentissage\\"+"Test_nbrsEntraiment_"+str(EPOCHS)+"_groupe_"+str(BS)+"_pourcentageAppInit_"+str(INIT_LR)+".txt"
    
    fichier=open(CheminFichier,"a")

    for CsvCour in CheminCSV:
        
        MonCSV=np.genfromtxt(CsvCour, delimiter=';')
        image = cv2.resize(MonCSV, (WIDTH, HEIGHT))
        image = image.astype("float") / 2300
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
    
        # classify the input image
        (autre, leve) = model.predict(image)[0]
    
        # build the label
        label = "levé_de_benne" if leve > autre else "autre"
        proba = leve if leve > autre else autre
        label = "{}: {:.2f}%".format(label, proba * 100)
        
        fichier.write("nom du csv :"+CsvCour+" "+label+"\n")
        #print("nom du csv :"+CsvCour+" "+label)
    
    fichier.close()
# draw the label on the image
#output = imutils.resize(orig, width=400) ne sert à rien pour un fichier csv
#output = cv2.imread(".\logo_exotic.jpg")
#cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
#	0.7, (0, 255, 0), 2)
#
## show the output image
#cv2.imshow("Output", output)
#cv2.waitKey(0)



