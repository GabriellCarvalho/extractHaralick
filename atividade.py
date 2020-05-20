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

data = []
data2 = []

# Funcao para extracao da característica
def extractHaralick(image):
    # extrai vetor caracteristico de haralick para os 4 tipos de adjacencia
    textures = mt.features.haralick(image)
    # tira a media e retorna o vetor
    ht_mean = textures.mean(axis=0)
    return ht_mean

# Carrega as imagens de saudaveis e extrai a característica 
for file in glob.glob("images/saudavel/*.jpg"):
	image = cv2.imread(file)
    # converte a imagem para grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	haralick = extractHaralick(gray)
	haralick = numpy.append(haralick, 0) #Faz o append da classe 0
	data.append(haralick)
    
# Carrega as imagens de doentes e extrai a característica
for file in glob.glob("images/doente/*.jpg"):
	image = cv2.imread(file)
    # converte a imagem para grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	haralick = extractHaralick(gray)
	haralick = numpy.append(haralick, 0) #Faz o append da classe 0
	data2.append(haralick)
    
# Insere num arquivo .csv e salva
savetxt("data.csv", data, delimiter=',')
savetxt("data2.csv", data2, delimiter=',') 