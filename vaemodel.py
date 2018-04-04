import numpy as np
import os
import keras
from keras import regularizers, losses
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Reshape, BatchNormalization, Softmax, Concatenate
from keras.utils import plot_model

class multiVAE:
	def __init__(self, sampleLen, numUnits, enc_denseLayerSizes, enc_denseLayerActivations, enc_dropouts, enc_batchnorms, dec_denseLayerSizes, dec_denseLayerActivations, dec_dropouts, dec_batchnorms):

		self.sampleLen = sampleLen
		self.numUnits = numUnits
		self.enc_denseLayerSizes = enc_denseLayerSizes
		self.enc_denseLayerActivations = enc_denseLayerActivations
		self.enc_dropouts = enc_dropouts
		self.enc_batchnorms = enc_batchnorms
		self.dec_denseLayerSizes = dec_denseLayerSizes
		self.dec_denseLayerActivations = dec_denseLayerActivations
		self.dec_dropouts = dec_dropouts
		self.dec_batchnorms = dec_batchnorms
		self.model = None


	def encoder(self, m_i):
		for l, act, drop, bn in zip(self.enc_denseLayerSizes, self.enc_denseLayerActivations, self.enc_dropouts, self.enc_batchnorms):
			m_i = Dense(l, activation=act)(m_i)
			m_i = Dropout(drop)(m_i)
			if(bn):
				m_i = BatchNormalization()(m_i)
		return m_i


	def createInputList(self):
		m = [None]*self.numUnits
		for i, _ in enumerate(m):
			m[i] = Input(shape = (self.sampleLen, ))
		return m

	def network(self):
		m = self.createInputList()
		y = [None]*self.numUnits
		#use list comprehension to make this better
		for i, m_i in enumerate(m):
			y[i] = self.encoder(m_i)

		z_in = Concatenate()(y)
		z_out = Dense(self.enc_denseLayerSizes[-1]*self.numUnits)(z_in)
		self.model = Model(inputs=m, outputs=z_out)
		print(self.model.summary())
		plot_model(self.model, to_file='multiVAE.png')






