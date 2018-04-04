import numpy as np
import os
import keras
from keras import regularizers, losses
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Reshape, BatchNormalization, Softmax, Concatenate
from keras.utils import plot_model
from vaemodel import *


sampleLen = 5
numUnits = 3
enc_denseLayerSizes = [20, 20]
enc_denseLayerActivations = ['relu', 'relu']
enc_dropouts = [0, 0]
enc_batchnorms = [False, False]
dec_denseLayerSizes = [20, 20]
dec_denseLayerActivations = ['relu', 'relu']
dec_dropouts = [0, 0]
dec_batchnorms = [False, False]

myModel = multiVAE(sampleLen, numUnits, enc_denseLayerSizes, enc_denseLayerActivations, enc_dropouts, enc_batchnorms, dec_denseLayerSizes, dec_denseLayerActivations, dec_dropouts, dec_batchnorms)

myModel.network()