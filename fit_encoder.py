''' Fit autoencoder using the keras library in tensorflow

Written by Kenneth Decker, kdecker8@gatech.edu
'''
from __future__ import print_function

import numpy as np
import numpy.random as npr
import scipy.linalg as la
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import time
import warnings
import copy

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy import interpolate
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow import keras

from base_classes import *


class FitEncoder(BaseClass, BaseAdaptiveSampling, BaseManifoldBackmap):
    ''' Fit ROM using autoencoder neural network

    --- Inputs ---

    --- Inherits ---

    --- Attributes ---

    --- Methods ---

    --- References ---

    '''
    def __init__(self, hidden_layer_sizes=(100,), n_components=2,
            activation='relu', solver='adam', loss='mean_squared_error',
            interp_method='rbf', interp_kernel='thin_plate',
            interp_parameter=None, alpha=1e-4, batch_size=None,
            learning_rate_init=1e-3, learning_rate='adaptive', epochs=100,
            shuffle=True, tol=1e-8, metrics=False, cond=False, timing=True,
            random_state=None, scale_inputs=True, scale_outputs=True,
            scale_embedding=True, method='auto', backmap_method='auto',
            backmap_parameters=None, validation_fraction=0.2):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_components = n_components
        self.activation = activation
        self.solver = solver
        self.loss = loss
        self.interp_method = interp_method
        self.interp_kernel = interp_kernel
        self.interp_parameter = interp_parameter
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.shuffle = shuffle
        self.tol = tol
        self.metrics = metrics
        self.cond = cond
        self.timing = timing
        self.random_state = random_state
        self.scale_inputs = scale_inputs
        self.scale_outputs = scale_outputs
        self.scale_embedding = scale_embedding
        self.method = method
        self.backmap_method = backmap_method
        self.backmap_parameters = backmap_parameters
        self.validation_fraction = validation_fraction

    def execute(self):
        ''' Train model and set up prediction function. Run diagnostices
            based on values defined in settings
        '''

        # Initialize embedding object based on user defined method
        if self.timing:
            fit_start = time.time()

        self._fit_reduction()

        # Fit interpolation model
        if hasattr(self, 'training_params'):
            self._fit_interpolation()

        # Set up backmapping
        self._setup_backmap()

        # Set up nearest nerighbor computation
        if self.backmap_method == 'manifold':
            self.embedding_nbrs = \
                            NearestNeighbors(n_neighbors=self.k_neighbors)
            self.embedding_nbrs.fit(self.embedding)

        # Stop timer and print results ot the console
        if self.timing:
            fit_end = time.time()        
            print(self.method + ' training time = %f s' 
                                        % (fit_end - fit_start))

        # # Compute diagnostics
        # if self.metrics:
        #     self._compute_manifold_diagnostics()

    def predict(self, prediction_params, return_weights=False):
        ''' Make predictions at user-defined points

        --- Inputs ---
        prediction_params : (n_prediction, n_features) numpy.ndarray
            Array of points at which predicitons are made
        return_weights : bool (default False)
            If True, return reconstruction weights of prediction

        --- Returns ---
        predictions : (n_prediction, n_grid) numpy.ndarray
            Field predictions at input points
        weights : (n_predictions, k_neighbors) numpy.ndarray
            Array of reconstruction weights at each prediction point
        '''
        # Check shape of input array
        n_predictions, n_params = prediction_params.shape
        if n_params != self.n_training_params:
            raise ValueError('Number of prediction parameters must match \
                                number of training parameters')

        # Scale prediction parameters
        if self.scale_inputs:
            prediction_params = self.inputScaler.transform(prediction_params)

        # Evaluate interpolation functions
        interpolated_y_values = self._interp_function(prediction_params)
        if self.n_components == 1:
            interpolated_y_values = np.reshape(interpolated_y_values, (-1,1))

        # Compute reconstruction
        predictions = np.zeros((n_predictions, self.grid_dimension))
        for i in range(n_predictions):
            y_predict = interpolated_y_values[i,:]
            y_predict = np.reshape(y_predict, (1,-1))   

            recon = self.backmap_function(y_predict)
            recon = np.reshape(recon, (1,-1))

            if self.scale_outputs:
                recon = self.outputScaler.inverse_transform(recon)

            predictions[i,:] = recon.flatten()

        return predictions

    def _fit_reduction(self):
        ''' Setup model architecture using Keras
        '''
        num_layers = len(self.hidden_layer_sizes)
        input_layer = Input(shape=(self.grid_dimension,),
                            batch_size=self.batch_size)
        center_layer = Input(shape=(self.n_components,),
                            batch_size=self.batch_size)

        if self.method == 'auto':
            for i in range(num_layers):
                if i == 0:
                    encoded = Dense(self.hidden_layer_sizes[i], 
                            activation=self.activation)(input_layer)

                else:
                    encoded = Dense(self.hidden_layer_sizes[i], 
                            activation=self.activation)(encoded)

            encoded = Dense(self.n_components,
                            activation=self.activation)(encoded)

            for i in range(num_layers):
                if i == 0:
                    decoded = Dense(self.hidden_layer_sizes[i], 
                            activation=self.activation)(encoded)
                else:
                    decoded = Dense(self.hidden_layer_sizes[i], 
                            activation=self.activation)(decoded)

            decoded = Dense(self.grid_dimension, 
                                activation=self.activation,
                                use_bias=True)(decoded)

            autoencoder = Model(input_layer, decoded)

            autoencoder.compile(optimizer=self.solver, loss=self.loss,
                            metrics=['accuracy', 'mse'])
            autoencoder.fit(self.snapshots, self.snapshots, epochs=self.epochs,
                            batch_size=self.batch_size, shuffle=self.shuffle,
                            validation_split=self.validation_fraction)

            encoder = Model(autoencoder.input, 
                            autoencoder.layers[num_layers+1].output)

            for i in range(num_layers+1):
                if i == 0:
                    deco = autoencoder.layers[-1-num_layers](center_layer)
                else:
                    deco = autoencoder.layers[-1-num_layers+i](deco)

            decoder = Model(center_layer, deco)

            self.autoencoder = autoencoder
            self.encoder = encoder
            self.decoder = decoder
            self.embedding = encoder.predict(self.snapshots)

    def _setup_backmap(self):
        ''' Set up backmap
        '''
        if self.backmap_method == 'auto':
            self.backmap_function = self.decoder.predict

        elif self.backmap_method == 'manifold':
            self.eps = self.backmap_parameters.get('eps', 0.01)
            self.p = self.backmap_parameters.get('p', 4)
            self.cond = self.backmap_parameters.get('cond', False)
            self.backmap_function = self._compute_manifold_backmap

        elif self.backmap_method == 'gauss':
            l = np.mean(self.embedding.max(axis=0) 
                - self.embedding.min(axis=0))/float(self.num_snapshots)

            kernel = gpk.Matern(length_scale=l)
            self.backmap_regressor = GaussianProcessRegressor(
                                            kernel=kernel, alpha=1e-4)
            self.backmap_regressor.fit(self.embedding, self.snapshots)
            self.backmap_function = self.backmap_regressor.predict

        elif self.backmap_method == 'nn':
            self.backmap_regressor = MLPRegressor(**self.backmap_parameters)

            if self.scale_inputs and self.scale_embedding:
                self.backmap_regressor.fit(self.embedding, self.snapshots)
                self.backmap_function = self.backmap_regressor.predict

            elif self.scale_embedding:
                self.nn_output_scaler = MinMaxScaler()
                self.nn_output_scaler.fit(self.snapshots)
                self.backmap_regressor.fit(self.embedding, 
                        self.nn_output_scaler.transform(self.snapshots))

                if self.n_components == 1:
                    self.backmap_function = lambda X: \
                            self.nn_output_scaler.inverse_transform(
                                np.reshape(self.backmap_regressor.predict(X), 
                                    (1,-1)))
                else:
                    self.backmap_function = lambda X: \
                            self.nn_output_scaler.inverse_transform(
                                    self.backmap_regressor.predict(X))

            elif self.scale_outputs:
                self.nn_embedding_scaler = MinMaxScaler()
                self.nn_embedding_scaler.fit(self.embedding)
                self.backmap_regressor.fit(
                    self.nn_embedding_scaler.transform(self.embedding),
                    self.snapshots)

                self.backmap_function = lambda X: \
                        self.backmap_regressor.predict(
                            self.nn_embedding_scaler.transform(X))

            else:
                self.nn_embedding_scaler = MinMaxScaler()
                self.nn_embedding_scaler.fit(self.embedding)
                self.nn_output_scaler = MinMaxScaler()
                self.nn_output_scaler.fit(self.snapshots)

                self.backmap_regressor.fit(
                    self.nn_embedding_scaler.transform(self.embedding),
                    self.nn_output_scaler.transform(self.snapshots))

                if self.n_components == 1:
                    self.backmap_function = lambda X: \
                        self.nn_output_scaler.inverse_transform(
                            np.reshape(self.backmap_regressor.predict(
                                self.nn_embedding_scaler.transform(X)), (1,-1)))
                else:
                    self.backmap_function = lambda X: \
                        self.nn_output_scaler.inverse_transform(
                            self.backmap_regressor.predict(
                                self.nn_embedding_scaler.transform(X)))
