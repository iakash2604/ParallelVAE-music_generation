import numpy as np
import os
import keras
from keras import regularizers, losses
from keras.models import Sequential, Model
from keras.layers import Lambda, Input, Dense, Dropout, Reshape, BatchNormalization, Softmax, Concatenate
from keras.utils import plot_model
import keras.backend as K

class multiVAE:
	def __init__(self, sampleLen, numUnits, enc_denseLayerSizes, enc_denseLayerActivations, enc_dropouts, enc_batchnorms, dec_denseLayerSizes, dec_denseLayerActivations, dec_dropouts, dec_batchnorms, inf_layerSize):

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
		self.inf_layerSize = inf_layerSize
		self.trainModel = None
		self.inf_layer = None
		self.genModel = [None]*self.numUnits

	def sample_z(self, args):
	    mean, log_sigma = args
	    eps = K.random_normal(shape=(32, self.inf_layerSize), mean=0., stddev=1.)
	    return mean + K.exp(log_sigma / 2) * eps

	def createInputList(self):
		m = [None]*self.numUnits
		for i, _ in enumerate(m):
			m[i] = Input(shape = (self.sampleLen, ), name = 'input'+str(i+1))
		return m

	def encoder(self, m_i, i):
		temp = len(self.enc_dropouts)
		for l, act, drop, bn, j in zip(self.enc_denseLayerSizes, self.enc_denseLayerActivations, self.enc_dropouts, self.enc_batchnorms, range(temp)):
			m_i = Dense(l, activation=act, name=str(i+1)+'enc_dense'+str(j+1))(m_i)
			m_i = Dropout(drop, name=str(i+1)+'enc_dropout'+str(j+1))(m_i)
			if(bn):
				m_i = BatchNormalization(name=str(i+1)+'enc_batchnorm'+str(j+1))(m_i)
		return m_i

	def decoder(self, z_i, i):
		temp = len(self.dec_dropouts)
		for l, act, drop, bn, j in zip(self.dec_denseLayerSizes, self.dec_denseLayerActivations, self.dec_dropouts, self.dec_batchnorms, range(temp)):
			z_i = Dense(l, activation=act, name=str(i+1)+'dec_dense'+str(j+1))(z_i)
			z_i = Dropout(drop, name=str(i+1)+'dec_dropout'+str(j+1))(z_i)
			if(bn):
				z_i = BatchNormalization(name=str(i+1)+'dec_batchnorm'+str(j+1))(z_i)
		return z_i

	def createFullNetwork(self):
		m = self.createInputList()
		y = [None]*self.numUnits
		m_ = [None]*self.numUnits
		#use list comprehension to make this better
		for i, m_i in enumerate(m):
			y[i] = self.encoder(m_i, i)

		# y_len = y[0].get_shape()[1:].as_list()[0]
		z_in = Concatenate()(y)
		mean = Dense(self.inf_layerSize, activation='linear', name='mean')(z_in)
		log_sigma = Dense(self.inf_layerSize, activation='linear', name='stddev')(z_in)
		z_out = Lambda(self.sample_z, name='inf_layer')([mean, log_sigma])
		
		# self.inf_layer = Input(shape = z_out.get_shape().as_list())
		# self.createDecoderModel(z_out)
		
		for i in range(self.numUnits):
			m_[i] = self.decoder(z_out, i)

		self.trainModel = Model(inputs=m, outputs=m_)
		plot_model(self.trainModel, to_file='multiVAE.png')

	def createGenModel(self):
		temp = len(self.dec_batchnorms)
		self.inf_layer = Input(shape = self.trainModel.get_layer('inf_layer').output_shape, name='gen_input')
		for i in range(self.numUnits):
			temp_layer = self.inf_layer
			for j, bn in zip(range(temp), self.dec_batchnorms):
				temp_layer = self.trainModel.get_layer(str(i+1)+'dec_dense'+str(j+1))(temp_layer)
				temp_layer = self.trainModel.get_layer(str(i+1)+'dec_dropout'+str(j+1))(temp_layer)
				if(bn):
					temp_layer = self.trainModel.get_layer(str(i+1)+'dec_batchnorm'+str(j+1))(temp_layer)

			self.genModel[i] = Model(inputs=self.inf_layer, outputs=temp_layer)
			plot_model(self.genModel[i], to_file='genModel'+str(i+1)+'.png')
		