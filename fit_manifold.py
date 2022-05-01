''' Fit manifold learning object using utilities in scikit learn

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


class FitManifold(BaseClass, BaseManifoldBackmap, BaseAdaptiveSampling):
    ''' Fit ROM using manifold learning

    --- Inputs ---
    method : str (default 'isomap')
        Manifold learning method [isomap, lle, hessian, modified, ltsa, le]
    k_neighbors : int (default 5)
        Number of nearest neighbors used to build graph. If 
        k_neighbors==-1, value of k is computed by optimization
    n_components : int (default 2)
        Dimension of the manifold
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
    backmap_method : str (default 'manifold')
        Name of method to carry our backmapping ['manifold', 'nn', 'gauss']
    backmap_parameters : dict (default None)
        Backmapping parameters, names and values are method specific
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
    n_jobs: int (default None)
        Number of parallel jobs to run. If None, defaults to 1

    --- Inherits ---
    base_classes.BaseClass, base_classes.BaseManifoldBackmap,
    base_classes.BaseAdaptiveSampling

    --- Attributes ---

    --- Methods ---

    --- References ---
    Franz, T., "Reduced-Order Modleing for Steady Transonic Flows via 
        Manifold Learning," Ph. D. Thesis, Deutches Zentrum fur Luftund
        Raumfahrt, 2016

    '''
    def __init__(self, method='isomap', k_neighbors=5, n_components=2,
            interp_method='rbf', interp_kernel='thin_plate',  
            interp_parameter=None, backmap_method='manifold', 
            backmap_parameters=None, metrics=False, timing=True, 
            random_state=None, scale_inputs=False, scale_outputs=False, 
            scale_embedding=False, n_jobs=None):
        self.method = method
        self.k_neighbors = k_neighbors
        self.n_components = n_components
        self.interp_method = interp_method
        self.interp_kernel = interp_kernel
        self.interp_parameter = interp_parameter
        self.backmap_method = backmap_method
        self.backmap_parameters = backmap_parameters
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

        if self.k_neighbors == -1:
            self._compute_optimal_k()
        else:
            self._fit_reduction()

        # Fit interpolation model
        if hasattr(self, 'training_params'):
            self._fit_interpolation()

        # Set up backmapping
        self._setup_backmap()

        # Bound weights for manifold backmapping
        if self.backmap_method == 'manifold' \
        and self.backmap_parameters['bound_weights']:
            self._bound_weights(
                self.backmap_parameters['bound_weights_params'])

        # Stop timer and print results ot the console
        if self.timing:
            fit_end = time.time()
            print(self.method + ' training time = %f s' 
                                        % (fit_end - fit_start))

        # Compute diagnostics
        if self.metrics:
            self._compute_reduction_diagnostics()

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
        interpolated_y_values = np.atleast_2d(interpolated_y_values)

        # Compute reconstruction
        predictions = np.zeros((n_predictions, self.grid_dimension))
        w_vec = np.zeros((self.k_neighbors, n_predictions))
        for i in range(n_predictions):
            y_predict = interpolated_y_values[i,:]
            y_predict = np.reshape(y_predict, (1,-1))   

            if return_weights and self.backmap_method == 'manifold':
                w_vec[:,i], recon = self.backmap_function(
                                            y_predict, return_weights=True)
            else:
                recon = self.backmap_function(y_predict)

            recon = np.reshape(recon, (1,-1))

            if self.scale_outputs:
                recon = self.outputScaler.inverse_transform(recon)

            predictions[i,:] = recon.flatten()

        if return_weights:
            return (predictions, w_vec)
        else:
            return predictions

    def _fit_reduction(self):
        ''' Generate manifold learning object, fit model, and store
        '''
        if self.method == 'isomap':
            self.fit = manifold.Isomap(
                n_neighbors=self.k_neighbors,
                n_components=self.n_components, 
                eigen_solver='dense',
                n_jobs=self.n_jobs
            )
            self.fit.fit(self.snapshots)
            self.data_nbrs = self.fit.nbrs_
        
        elif self.method == 'LLE' or self.method == 'lle':
            self.fit = manifold.LocallyLinearEmbedding(
                n_neighbors=self.k_neighbors, 
                n_components=self.n_components, 
                method='standard', 
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            self.fit.fit(self.snapshots)
            self.data_nbrs = self.fit.nbrs_
        
        elif self.method == 'hessian' or self.method == 'hess'\
        or self.method == 'hlle' or self.method == 'HLLE':
            self.fit = manifold.LocallyLinearEmbedding(
                n_neighbors=self.k_neighbors,
                n_components=self.n_components, 
                method='hessian', 
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            self.fit.fit(self.snapshots)
            self.data_nbrs = self.fit.nbrs_

        elif self.method == 'modified' \
        or self.method == 'mlle' or self.method == 'MLLE':
            self.fit = manifold.LocallyLinearEmbedding(
                n_neighbors=self.k_neighbors,
                n_components=self.n_components, 
                method='modified', 
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            self.fit.fit(self.snapshots)
            self.data_nbrs = self.fit.nbrs_

        elif self.method == 'LTSA' or self.method == 'ltsa':
            self.fit = manifold.LocallyLinearEmbedding(
                n_neighbors=self.k_neighbors,
                n_components=self.n_components, 
                method='ltsa', 
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            self.fit.fit(self.snapshots)
            self.data_nbrs = self.fit.nbrs_

        elif self.method == 'LE' or self.method == 'laplacian'\
        or self.method == 'le' or self.method == 'spectral':
            self.fit = manifold.SpectralEmbedding(
                n_neighbors=self.k_neighbors,
                n_components=self.n_components,
                affinity='rbf',
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            self.fit.fit(self.snapshots)
            self.data_nbrs = NearestNeighbors(n_neighbors=self.k_neighbors)
            self.data_nbrs.fit(self.snapshots)

        if self.scale_embedding:
            self.embedding_scaler = MinMaxScaler()
            self.embedding_scaler.fit(self.fit.embedding_)
            self.embedding = \
                        self.embedding_scaler.transform(self.fit.embedding_)
        else:
            self.embedding_scaler = None
            self.embedding = self.fit.embedding_

    def _compute_optimal_k(self, metric='best'):
        ''' Implement algorithm for computing the optimal value of k by 
            minimizing Kruskal's stress or the variance of distance ratios

        --- Inputs ---
        metric : str (default 'best')
            Objective function to use when optimizing k. If best, use 
            Kruskal's stress for Isomap and variance of distance ratios
            otherwise ['best', 'kruskal', 'sammon', 'resid', 'variance'] 
        '''
        # Initialize list of possible k values 
        if self.method == 'hessian' or self.method == 'hess'\
        or self.method == 'hlle' or self.method == 'HLLE':
            k_list = list(range(self.n_components*(self.n_components+3)//2 + 1,
                                self.num_snapshots-1))
        else:
            k_list = list(range(self.n_components+1, self.num_snapshots//2))

        s_opt = np.inf
        reduction_calls = 0
        k_history = {}

        # Chose optimization objective
        if metric == 'best':
            if self.method == 'isomap': 
                metric = 'kruskal'
            else:
                metric = 'variance'

        # Find optimal k value using bisection algorithm
        while len(k_list) > 1:
            if len(k_list) <= 3:
                si = [0]*len(k_list)
                for i, k in enumerate(k_list):
                    if k in k_history.keys():
                        si[i] = k_history[k]

                    else:
                        self.k_neighbors = k
                        self._fit_reduction()
                        self._compute_reduction_diagnostics()
                        reduction_calls += 1

                        if metric == 'kruskal':
                            si[i] = self.k_stress
                        elif metric == 'sammon':
                            si[i] = self.s_stress
                        elif metric == 'resid':
                            si[i] = self.residual_variance
                        elif metric == 'variance':
                            si[i] = self.ratio_variance

                        k_history.update({k : si[i]})

                k_list = [k_list[si.index(min(si))]]

            else:
                si = [0]*3
                for i, k_ind in enumerate([int(len(k_list)/4), 
                                        int(len(k_list)/2), 
                                        int(3*(len(k_list)/4))]):

                    if k_list[k_ind] in k_history.keys():
                        si[i] = k_history[k_list[k_ind]]

                    else:
                        self.k_neighbors = k_list[k_ind]
                        self._fit_reduction()
                        self._compute_reduction_diagnostics()
                        reduction_calls += 1

                        if metric == 'kruskal':
                            si[i] = self.k_stress
                        elif metric == 'sammon':
                            si[i] = self.s_stress
                        elif metric == 'resid':
                            si[i] = self.residual_variance
                        elif metric == 'variance':
                            si[i] = self.ratio_variance

                        k_history.update({k_list[k_ind] : si[i]})

                if min(si) == si[0]:
                    s_opt = si[0]
                    del k_list[int(len(k_list)/2):]
                elif min(si) == si[1]:
                    s_opt = si[1]
                    del k_list[int(3*len(k_list)/4):]
                    del k_list[0:int(len(k_list)/4)]
                else:
                    del k_list[0:int(len(k_list)/2)]

        self.k_neighbors = k_list[0]
        self._fit_reduction()
        reduction_calls += 1

        print('%s optimal k = %d' % (self.method, self.k_neighbors))
        print('Total dimension reduction calls = %d' % reduction_calls)

    def _bound_weights(self, validation_params, # FIX
                                    w_bounds=(-0.25, 1.25), max_iter=20):
        ''' Check if reconstruction points lie within bounds at interp points
            If not, increase k until all weights lie within bounds or until
            max k is reached
        
        --- Inputs ---
        validation_params : (n_predictions, n_params) numpy.ndarray
            Array of points at which predicitons are made
        w_bounds : tuple (default (-0.25, 1.25))
            Min and max value
            
            s of reconstruction weights for
        max_iter : int (default 10)
            Maximum number of iterations for bounding weights
        '''
        # Check reconstruction weights for optimal k
        _, w_array = self.predict(validation_params, return_weights=True)
        w_max = w_array.max(axis=None)
        w_min = w_array.min(axis=None)

        # if best k does not meet weight constraint, increase k
        k_best = 0
        sum_best = np.inf
        if w_max > w_bounds[1] or w_min < w_bounds[0]:

            count = 0
            while w_max > w_bounds[1] or w_min < w_bounds[0]:
                # Increment k and re-fit model
                self.k_neighbors += 1
                self._fit_reduction()
                self._fit_interpolation()
                self._setup_backmap()
                
                _, w_array = self.predict(validation_params, 
                                                return_weights=True)
                w_max = w_array.max(axis=None)
                w_min = w_array.min(axis=None)

                # Store best model based on sum of worst weight coefficients
                if (w_max - w_min) < sum_best:
                    k_best = self.k_neighbors
                    sum_best = w_max - w_min
                    w_min_best = w_min
                    w_max_best = w_max

                # Check max iter, if exceeded, re-fit wiht best k and stop
                count += 1
                if count == max_iter \
                or self.k_neighbors == (self.num_snapshots-1):
                    self.k_neighbors = k_best
                    self._fit_reduction()
                    self._fit_interpolation()
                    self._setup_backmap()
                    print('Max k reached')
                    print('k_best   = %d' % k_best)
                    print('w_min    = %f' % w_min_best)
                    print('w_max    = %f' % w_max_best)
                    break

            if self.metrics:
                self._compute_reduction_diagnostics()

            # Print updated value of k if any
            print('New k after weight bounding = %d' % self.k_neighbors)

    def _setup_backmap(self):
        ''' Set up backmapping
        '''
        # Construct Nearest Neighbors graph in the membedding space
        self.embedding_nbrs = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.embedding_nbrs.fit(self.embedding)

        # Set up backmap function 
        if self.backmap_method == 'manifold':
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
