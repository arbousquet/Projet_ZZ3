# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 18:14:34 2019

@author: abousquet
"""
from Train import EntrainementReseau
from test_reseau_csv import TestReseau
from imutils import paths


#paramètres pour l'entrainemetent du réseau

#nbrs d'itération
MAX_EPOCHS = 70
MIN_EPOCHS = 5
#nbrs d'exemples par itération
MAX_BS = 20
MIN_BS = 5

#% apprentissage initial
INIT_LR = 1e-3

#dimension fichier csv
WIDTH=3
HEIGHT=600


CheminEntrainementCSV=sorted(list(paths.list_files(".\Data",".csv") ) )
CheminTestCSV=sorted(list(paths.list_files(".\\Validation",".csv") ) )

for epochs in range (MIN_EPOCHS,MAX_EPOCHS,5):
    for BatchSize in range (MIN_BS,MAX_BS,5):
        print(" Début de l'entrainement/test/validation  avec les params : Epochs="+str(epochs)+" Batch size="+str(BatchSize))
        #entrainer le réseau avec les params d'epochs et de batch courant
        EntrainementReseau(CheminEntrainementCSV,epochs,BatchSize,INIT_LR,WIDTH,HEIGHT)
        #tester le modèle sur la base
        TestReseau(CheminTestCSV,epochs,BatchSize,INIT_LR,WIDTH, HEIGHT)
        
        
        
print("FIN DU PROGRAMME")    