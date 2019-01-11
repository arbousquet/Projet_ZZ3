# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 19:29:58 2018

@author: abousquet
"""

import csv
from random import randint


"""
generation bdd Entrainement
"""


#for i in range(0,500):
#    fichier=open(".\\Entrainement\\leve_benne\\"+str(i)+".csv","a")
#    """
#    On va de 0 à 27 car la taille des images virtulles finales doit être de 28X28X3
#    """
#    for j in range(0,27):
#        fichier.write(str( randint(0,100) )+","+str( randint(0,100) )+","+str( randint(0,100) )+"\n")
#    
#    fichier.close()
#
#
#
#
#
#for i in range(0,500):
#    fichier=open(".\\Entrainement\\autre\\"+str(i)+".csv","a")
#    
#    """
#    On va de 0 à 27 car la taille des images virtulles finales doit être de 28X28X3
#    """
#    for j in range(0,27):
#        fichier.write(str( randint(150,1000) )+","+str( randint(150,1000) )+","+str( randint(150,1000) )+"\n")
#    
#    fichier.close()



"""
Génération bdd de validation
"""
for i in range(11,20):
    fichier=open(".\\validation\\"+str(i)+".csv","a")
    
    """
    On va de 0 à 27 car la taille des images virtulles finales doit être de 28X28X3
    """
    for j in range(0,27):
        fichier.write(str( randint(150,1000) )+","+str( randint(150,1000) )+","+str( randint(150,1000) )+"\n")
    
    fichier.close()

"""
vérification de la BDD
"""

fichier=open(".\\validation\\1.csv","r")
reader = csv.reader(fichier)

for row in reader:
    print('\t'.join(row))

fichier.close()
