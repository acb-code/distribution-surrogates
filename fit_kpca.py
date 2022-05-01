''' Fit kernel pca object using utilities in scikit learn

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

from scipy.spatial import Delaunay
from scipy.optimize import minimize, basinhopping
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, decomposition
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
# from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.utils.graph import single_source_shortest_path_length as graph_shortest_path
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as gpk
from sklearn.neural_network import MLPRegressor
from scipy import interpolate

from base_classes import *


class FitKPCA(BaseClass, BaseManifoldBackmap, BaseAdaptiveSampling):
    ''' Fit ROM using kernel principal component analysis (KPCA)

    --- Inputs ---
    method : str (default 'rbf')
        Kernel function to use for reduction, ['linear', 'poly', 'rbf',
        'sigmoid', 'cosine']. Can also pass custom kernel object generated
        using sklearn.gaussian_process.kernels module
    n_components : int (default None)
        Number of components to preserve. If None, reduction will 
        preserve all possible components
    interp_method : str (default 'rbf')
        Interpolator class to use, currently supports radial basis 
        function interpolator and gaussian process regressor 
        ['rbf', 'gauss', 'nn']
    interp_kernel: str (default 'thin_plate')
        Kernel function to use for interpolation. 
        If interp_method=='rbf', interp_kernel = ['multiquadric', 'cubic', 
            'gaussian','inverse_multiquadric', 'linear', 'quintic', 
            'thin_plate']
        If interp_method=='gauss', interp_kernel = ['constant', 'sine', 
            'dot', 'rbf', 'matern', 'quad'] or instance of 
            sklearn.gaussian_process.kernels.Kernel
        If interp_method=='nn', value is ignored
    interp_parameter : float or dict (default None)
        Paremeter used to train interpolator. None defers to scipy or
        scikit learn defaults or pre-stated heuristic
    backmap_method : str (default 'auto')
        Name of method to carry our backmapping 
        ['manifold', 'nn', 'gauss', 'auto']
    backmap_parameters : str (default None)
        Backmapping parameters, names and values are method specific
    alpha : float (default 1e-6)
        Regularization parameter for kernel pre-imaging
    metrics : bool (default False)
        If true, compute manifold learning diagnostics
    timing : bool (default True)
        If True, track model fit and evaluation time
    random_state : int (default None)
        Integer to be used as random seed for training model subsets
    scale_inputs : bool (default False)
        If True, scale input params to the interval [0,1]
    scale_outputs : bool (default False)
        If True, scale output values to the interval [0,1]
    scale_embedding : bool (default False)
        If True, scale embedding values to the interval [0,1]

    --- Inherits ---

    --- Attributes ---

    --- Methods ---

    --- References ---

    '''
    def __init__(self, method='rbf', n_components=None,
            interp_method='rbf', interp_kernel='thin_plate',  
            interp_parameter=None, backmap_method='auto', 
            backmap_parameters=None, alpha=1e-6, metrics=False, timing=True,
            random_state=None, scale_inputs=False, scale_outputs=False, 
            scale_embedding=False, n_jobs=None):
        self.method = method
        self.n_components = n_components
        self.interp_method = interp_method
        self.interp_kernel = interp_kernel
        self.interp_parameter = interp_parameter
        self.backmap_method = backmap_method
        self.backmap_parameters = backmap_parameters
        self.alpha = alpha
        self.metrics = metrics
        self.timing = timing
        self.random_state = random_state
        self.scale_inputs = scale_inputs
        self.scale_outputs = scale_outputs
        self.scale_embedding = scale_embedding
        self.n_jobs = n_jobs

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

        # Stop timer and print results ot the console
        if self.timing:
            fit_end = time.time()
            print(self.method + ' training time = %f s' 
                                    % (fit_end - fit_start))

        # # Compute diagnostics
        # if self.metrics:
        #     self._compute_reduction_diagnostics()

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
        ''' Generate KPCA object, fit model, and store
        '''
        # get number of snapshots and intialize number of components
        num_snapshots, num_dim = np.shape(self.snapshots)

        self.fit = decomposition.KernelPCA(
            n_components=self.n_components,
            kernel=self.method, 
            alpha=self.alpha, 
            fit_inverse_transform=True,
            eigen_solver='dense',
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.fit.fit(self.snapshots)
        if self.n_components is None:
            self.n_components = self.fit.lambdas_.size

        if self.scale_embedding:
            self.embedding_scaler = MinMaxScaler()
            self.embedding_scaler.fit(self.fit.transform(self.snapshots))
            self.embedding = self.embedding_scaler.transform(
                                        self.fit.transform(self.snapshots))
        else:
            self.embedding_scaler = None
            self.embedding = self.fit.transform(self.snapshots)

    def _setup_backmap(self):
        ''' Set up backmap
        '''
        if self.backmap_method == 'auto':
            self.backmap_function = self.fit.inverse_transform

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









