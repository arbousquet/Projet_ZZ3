    # -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:26:33 2018

@author: abousquet
"""

import csv
import numpy as np

CheminCSV=""

MonCSV=np.genfromtxt(CheminCSV, delimiter=',')#tableau à 4 colones (x,y,z,timestamp) et 86400 ligne

NbrsLigne=len(MonCSV)
i=0
trouve=0
TableauDeMesures=[]
prediction=0

while i<NbrsLigne:
    
    if((i+30) >= NbrsLigne):
        j=NbrsLigne
    else:
        j=i+30
        
    AcquisitionInstantT=MonCSV[i]
    AcquisitionInstantT30=MonCSV[j]
    
    #s'il n'y a pas de variation notable entre l'un des trois axes
    if(AcquisitionInstantT[0] - AcquisitionInstantT[0]*0.1 <= AcquisitionInstantT30[0] <= AcquisitionInstantT[0] + AcquisitionInstantT[0]*0.1 or 
       AcquisitionInstantT[1] - AcquisitionInstantT[1]*0.1 <= AcquisitionInstantT30[1] <= AcquisitionInstantT[1] + AcquisitionInstantT[1]*0.1 or 
       AcquisitionInstantT[2] - AcquisitionInstantT[2]*0.1 <= AcquisitionInstantT30[2] <= AcquisitionInstantT[2] + AcquisitionInstantT[2]*0.1 ):
        i=i+30 #on avance dans le fichier
    else:
        #on a potentiellement trouvé une levé/monté de benne entre t et t+30s
        #on va mainteant affiner la recherche pour trouver le début de l évènement
        k=i
        arret=0
        while (k <=i+30 and arret==0):
            if(k+4 <NbrsLigne):
                AcquisitionInstantT1=MonCSV[k]
                AcquisitionInstantT2=MonCSV[k+1]
                AcquisitionInstantT3=MonCSV[k+2]
                AcquisitionInstantT4=MonCSV[k+3]
                AcquisitionInstantT5=MonCSV[k+4]
                
                moyenneX=(AcquisitionInstantT[0]+AcquisitionInstantT2[0]+AcquisitionInstantT3[0]+AcquisitionInstantT4[0]+AcquisitionInstantT5[0])/5
                moyenneY=(AcquisitionInstantT[1]+AcquisitionInstantT2[1]+AcquisitionInstantT3[1]+AcquisitionInstantT4[1]+AcquisitionInstantT5[1])/5
                moyenneZ=(AcquisitionInstantT[2]+AcquisitionInstantT2[2]+AcquisitionInstantT3[2]+AcquisitionInstantT4[2]+AcquisitionInstantT5[2])/5
                
                #Si la moyenne est "centrée" autour de l'aquisition initiale alors on considère qu'il ne se passe rien
                if(AcquisitionInstantT[0]-AcquisitionInstantT[0]*0.05<= moyenneX <= AcquisitionInstantT[0]+AcquisitionInstantT[0]*0.05 or 
                   AcquisitionInstantT[1]-AcquisitionInstantT[1]*0.05<= moyenneY <= AcquisitionInstantT[1]+AcquisitionInstantT[1]*0.05 or
                   AcquisitionInstantT[2]-AcquisitionInstantT[2]*0.05<= moyenneX <= AcquisitionInstantT[2]+AcquisitionInstantT[2]*0.05 ):
                    k=k+5
                else:#on est sûr d'avoir détecté quelque chose
                    #on lance le DL/ML à partir de l'acquisition T jusqu'a T+60s
                    ValMax=k+60
                    if(k+60>=NbrsLigne):
                        ValMax=NbrsLigne
                    #for i in range(1, N + 1):  
                    for l in range(k,ValMax):#for i in range (première va à prendre en compte, première val à NE PAS prendre en compte)
                        TableauDeMesures=TableauDeMesures+MonCSV[l,:3]#on selectionne les valeurs des trois axes sur une minute
                    
                    #Fonction pour le modèle de machine learning
                    #ne pas oublier de changer la valeur de prediction
                    #if prediction==1: un mvt de benne a été détecté
                    #   mettre le timestamp de AcquisitionInstantT1 dans un tableau 
                
            else:
                arret=1
                
                
                
        
        