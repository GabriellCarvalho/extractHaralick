# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:31:14 2020

@author: gabriel
"""

import cv2
import mahotas as mt
import numpy 
from numpy import savetxt
import glob


# Funcao para extracao da característica
def extractHaralick(image):
    # extrai vetor caracteristico de haralick para os 4 tipos de adjacencia
    textures = mt.features.haralick(image)
    # tira a media e retorna o vetor
    ht_mean = textures.mean(axis=0)
    return ht_mean

# Funcao para carregar as imagens e extrair as características 
def setData(caminho):
    data =[]
    for file in glob.glob(caminho + "/*.jpg"):
        image = cv2.imread(file)
        # converte a imagem para grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haralick = extractHaralick(gray)
        haralick = numpy.append(haralick, 0) #Faz o append da classe 0
        data.append(haralick)
    return data

dataSaudavel = setData("images/saudavel")
dataDoente = setData("images/doente")
savetxt("dataSaudavel.csv", dataSaudavel+dataDoente, delimiter=',')
