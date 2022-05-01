''' Fit multifidelity PCA ROM using manifold alignment

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
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as gpk
from sklearn.neural_network import MLPRegressor
from scipy import interpolate

from base_classes import *


class FitMultifidelityKPCA(BaseAlignment):
    ''' Fit ROM using principal component analysis (PCA)

    --- Inputs ---
    method: str (default 'rbf')
        Kernel function to use for reduction ['linear', 'poly', 'rbf',
        'sigmoid', 'cosine']. Can also pass custom kernel object 
        generated using sklearn.gaussian_process.kernels module
    n_components: int (default None)
        Number of componentes to preserve. If None, reduction will 
        preserve all possible components by default
    random_state : int (default None)
        Integer to be used as random seed for eigensolver
    interp_method : str (default 'thin_plate')
        Interpolation method ['rbf', 'gauss', 'hk', 'nn']  
    interp_kernel: str (default 'thin_plate')
        Kernel function to use for interpolation. 
        If interp_method=='rbf', interp_kernel = ['multiquadric', 'cubic', 
            'gaussian','inverse_multiquadric', 'linear', 'quintic', 
            'thin_plate']
        If interp_method=='gauss' or 'hk', interp_kernel = ['constant', 
            'sine', 'dot', 'rbf', 'matern', 'quad'] or instance of 
            sklearn.gaussian_process.kernels.Kernel
        If interp_method=='nn' or 'hk', value is ignored
    interp_parameter : float (default None)
        Paremeter used to train RBF interpolation. None defers to scipy 
        default
    backmap_method : str (default 'auto')
        Name of the method to carry out backmapping
        ['manifold', 'nn', 'gauss', 'auto']
    backmap_parameters: dict (default None)
        Backmap parameters, names and values are method specific
    alpha: float (default (1e-6)
        Refularization parameter for pre-imaging. Only used if 
        backmap_method == 'auto'
    timing : bool (default)
        If True, track timing
    metrics : bool (default False)
        If true, compute manifold learning diagnostics
    random_state : int (default None)
        Integer to be used as random seed for training model subsets
    scale_inputs : bool (default False)
        If True, scale input params to the interval [0,1]
    scale_outputs : bool (default False)
        If True, scale output values to the interval [0,1]
    scale_embedding : bool (default False)
        If True, scale embedding values to the interval [0,1]
    n_jobs: int (default None)
        Number of threads used to perform reduction. If None, defaults
        to single thread

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
        ''' Execute the operations needed to carry out operation
        '''
        # Create embedding object
        if self.timing:
            fit_start = time.time()
        
        # Perform Dimensionality Reduction
        self._fit_reduction()

        # Align manifolds using Procrustes analysis
        self._align_manifolds()

        # Fit interpolation model and compute MFE at fit points
        if hasattr(self, 'params'):
            self._fit_interpolation()
        
        # Stop timer and print results to the console
        if self.timing:
            fit_end = time.time()
            print('KPCA Model fit time = %f s' % (fit_end - fit_start))

    def predict(self, prediction_params):
        ''' Make predictions at user-defined points

        --- Inputs ---
        prediction_params : (n_prediction, n_params) numpy.ndarray
            Array of points at which predicitons are made

        --- Returns ---
        predictions : (n_prediction, n_grid) numpy.ndarray
            Field predictions at input points
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

        # Compute reconstruction at each point
        predictions = np.zeros((n_predictions, self.grid_dimension))
        for i in range(n_predictions):
            y_predict = interpolated_y_values[i,:]
            y_predict = np.reshape(y_predict, (1,-1))   

            recon = self.hf_fit.inverse_transform(y_predict)
            recon = np.reshape(recon, (1,-1))

            if self.scale_outputs:
                recon = self.outputScaler.inverse_transform(recon)

            predictions[i,:] = recon.flatten()
    
        return predictions

    def _fit_reduction(self):
        ''' Generate PCA object, fit model, and store
        '''
        # Initialize lists
        self.lf_fits = []
        self.lf_embeddings = []
        if self.scale_embedding:
            self.lf_embedding_scalers = []
        else:
            self.lf_embedding_scalers = None

        # Begin by fitting PCA on high fidelity data 
        self.hf_fit = decomposition.KernelPCA(
            n_components=self.n_components,
            kernel=self.method,
            alpha=self.alpha,
            fit_inverse_transform=True,
            eigen_solver='dense',
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.hf_fit.fit(self.snapshots)
        if self.n_components is None:
            self.n_components = self.hf_fit.lambdas_.size

        # Store modes and embedding
        self.modes = self.hf_fit.alphas_
        if self.scale_embedding:
            self.embedding_scaler = MinMaxScaler()
            self.embedding_scaler.fit(self.hf_fit.transform(self.snapshots))
            self.embedding = self.embedding_scaler.transform(
                                        self.hf_fit.transform(self.snapshots))
        else:
            self.embedding_scaler = None
            self.embedding = self.hf_fit.transform(self.snapshots)

        # Fit each low fidelity data set
        for i in range(len(self.linked_lf_snapshots)):
            combined_data = np.vstack(
                                (self.linked_lf_snapshots[i],
                                 self.unlinked_lf_snapshots[i]))

            # Perform reduction and store
            lf_fit = decomposition.KernelPCA(
                n_components=self.n_components,
                kernel=self.method,
                alpha=self.alpha,
                fit_inverse_transform=True,
                eigen_solver='dense',
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            lf_fit.fit(combined_data)
            self.lf_fits.append(lf_fit)

            # Extract embedding 
            if self.scale_embedding:
                scaler = MinMaxScaler()
                scaler.fit(lf_fit.transform(combined_data))
                self.lf_embedding_scalers.append(scaler)
                self.lf_embeddings.append(
                            scaler.transform(lf_fit.transform(combined_data)))
            else:
                self.lf_embeddings.append(lf_fit.transform(combined_data))

