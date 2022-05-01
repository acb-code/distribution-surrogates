''' Fit multifidelity manifold learning ROM using manifold alignment

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


class FitMultifidelityManifold(BaseAlignment, BaseManifoldBackmap):
    ''' Fit ROM using autoencoder neural network

    --- Inputs ---
    method : str (default 'isomap')
        Manifold learning method [isomap, lle, hessian, modified, ltsa, le]
    k_neighbors : int or tuple of ints (default 5)
        Number of nearest neighbors used to build graph. If tuple, each 
        entry corresponds to each input data set. First entry applies to 
        high fidelity data set, each successive entry applies to each
        low fidelity data set in the order in which they were added
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
    backmap_parameters : str (default None)
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
        Number of jobs to run in parallel. If None, default to 1

    --- Inherits ---

    --- Attributes ---

    --- Methods ---

    --- References ---

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
        ''' Train model and set up prediction function. Run diagnostics
            based on values defined in settings
        '''
        # Initialize embedding object based on user defined method
        if self.timing:
            fit_start = time.time()

        # Perform Dimensionality Reduction
        if self.k_neighbors == -1:
            self._compute_optimal_k()
        else:
            self._fit_reduction()
        
        # Align manifolds using Procrustes analysis
        self._align_manifolds()

        # Fit interpolation model
        if hasattr(self, 'params'): 
            self._fit_interpolation()

        # Set up backmapping
        self._setup_backmap()

        # Bound weights for manifold backmapping
        if self.backmap_method == 'manifold' \
        and self.backmap_parameters['bound_weights']:
            self._bound_weights(
                self.backmap_parameters['bound_weights_params'])

        # Stop timer and print results to the console
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

    def _fit_single_reduction(self, data_set, k_neighbors):
        ''' Generate manifold learning object, fit model, and return 
            fit object

        --- Inputs ---
        data_set : (n_samples, n_grid) numpy.ndarray
            Set of snapshots to reduce using manifold learning
        k_neighbors : int
            Number of nearest neighbors for reduction

        --- Returns ---
        fit : sklearn.manifold instance
            Instance of manifold learning object from scikit learn
        '''
        if self.method == 'isomap':
            fit = manifold.Isomap(
                n_neighbors=k_neighbors,
                n_components=self.n_components, 
                eigen_solver='dense',
                n_jobs=self.n_jobs
            )
            fit.fit(data_set)
        
        elif self.method == 'LLE' or self.method == 'lle':
            fit = manifold.LocallyLinearEmbedding(
                n_neighbors=k_neighbors, 
                n_components=self.n_components, 
                method='standard', 
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            fit.fit(data_set)
        
        elif self.method == 'hessian' or self.method == 'hess' \
        or self.method == 'hlle' or self.method == 'HLLE':
            fit = manifold.LocallyLinearEmbedding(
                n_neighbors=k_neighbors,
                n_components=self.n_components, 
                method='hessian', 
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            fit.fit(data_set)

        elif self.method == 'modified' \
        or self.method == 'mlle' or self.method == 'MLLE':
            fit = manifold.LocallyLinearEmbedding(
                n_neighbors=k_neighbors,
                n_components=self.n_components, 
                method='modified', 
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            fit.fit(data_set)

        elif self.method == 'LTSA' or self.method == 'ltsa':
            fit = manifold.LocallyLinearEmbedding(
                n_neighbors=k_neighbors,
                n_components=self.n_components, 
                method='ltsa', 
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            fit.fit(data_set)

        elif self.method == 'LE' or self.method == 'laplacian' \
        or self.method == 'le' or self.method == 'spectral':
            fit = manifold.SpectralEmbedding(
                n_neighbors=k_neighbors,
                n_components=self.n_components,
                affinity='rbf',
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            fit.fit(data_set)

        return fit

    def _fit_reduction_test(self):
        ''' Reduce all data sets and store each fit object
        '''
        # Error check k values
        if type(self.k_neighbors) is tuple \
        and len(self.k_neighbors) != len(self.unlinked_lf_snapshots)+1:
            raise RuntimeError('Length of k_neighbors must match the \
                                number of data sets provided')

        # Initialize lists
        self.lf_fits = []
        self.lf_data_nbrs = []
        self.lf_embeddings = []
        self.lf_embedding_nbrs = []
        if self.scale_embedding:
            self.lf_embedding_scalers = []
        else:
            self.lf_embedding_scalers = None

        # Fit manifold for high fidelity data set
        #   Extract value of k
        if type(self.k_neighbors) is int:
            k = self.k_neighbors
        else:
            k = self.k_neighbors[0]

        #   Perform single reduction and store
        self.hf_fit = self._fit_single_reduction(self.snapshots, k)

        #   Generate NearestNeighbors object of data
        if isinstance(self.hf_fit, manifold.SpectralEmbedding):
            self.hf_data_nbrs = NearestNeighbors(n_neighbors=k)
            self.hf_data_nbrs.fit(self.snapshots)
        else:
            self.hf_data_nbrs = self.hf_fit.nbrs_

        #   Extract high fidelity embedding 
        if self.scale_embedding:
            self.embedding_scaler = MinMaxScaler()
            self.embedding_scaler.fit(self.hf_fit.embedding_)
            self.embedding = \
                    self.embedding_scaler.transform(self.hf_fit.embedding_)
        else:
            self.embedding = self.hf_fit.embedding_

        #   Fit Nearest Neighbors in embedding space
        self.embedding_nbrs = NearestNeighbors(n_neighbors=k)
        self.embedding_nbrs.fit(self.embedding)

        # Fit manifold for each low fidelity set
        for i in range(len(self.linked_lf_snapshots)):
            combined_data = np.vstack(
                                (self.linked_lf_snapshots[i], 
                                 self.unlinked_lf_snapshots[i])) 

            # Extract value of k
            if type(self.k_neighbors) is int:
                k = self.k_neighbors
            else:
                k = self.k_neighbors[i+1]

            # Perform reduction and store
            lf_fit = self._fit_single_reduction(
                            self.linked_lf_snapshots[i], k)
            self.lf_fits.append(lf_fit)

            # Project unlinked points onto manifold and combine
            unlinked_embedding = lf_fit.transform(
                                    self.unlinked_lf_snapshots[i])
            combined_embedding = np.vstack((lf_fit.embedding_,
                                            unlinked_embedding))

            # Generate NearestNeighbors object of data
            if isinstance(lf_fit, manifold.SpectralEmbedding):
                data_nbrs = NearestNeighbors(n_neighbors=k)
                data_nbrs.fit(combined_data)
                self.lf_data_nbrs.append(data_nbrs)

            else:
                self.lf_data_nbrs = lf_fit.nbrs_

            # Extract embedding 
            if self.scale_embedding:
                scaler = MinMaxScaler()
                scaler.fit(combined_embedding)
                self.lf_embedding_scalers.append(scaler)
                self.lf_embeddings.append(scaler.transform(combined_embedding))
            else:
                self.lf_embeddings.append(combined_embedding) 

            # Fit Nearest Neighbors in embedding space
            embedding_nbrs = NearestNeighbors(n_neighbors=k)
            embedding_nbrs.fit(self.lf_embeddings[i])
            self.lf_embedding_nbrs.append(embedding_nbrs)

        # Only store the value of k for the high fidelity data for backmapping
        if type(self.k_neighbors) is tuple:
            self.k_neighbors = self.k_neighbors[0]

    def _fit_reduction(self):
        ''' Reduce all data sets and store each fit object
        '''
        # Error check k values
        if type(self.k_neighbors) is tuple \
        and len(self.k_neighbors) != len(self.unlinked_lf_snapshots)+1:
            raise RuntimeError('Length of k_neighbors must match the \
                                number of data sets provided')

        # Initialize lists
        self.lf_fits = []
        self.lf_data_nbrs = []
        self.lf_embeddings = []
        self.lf_embedding_nbrs = []
        if self.scale_embedding:
            self.lf_embedding_scalers = []
        else:
            self.lf_embedding_scalers = None

        # Fit manifold for high fidelity data set
        #   Extract value of k
        if type(self.k_neighbors) is int:
            k = self.k_neighbors
        else:
            k = self.k_neighbors[0]

        #   Perform single reduction and store
        self.hf_fit = self._fit_single_reduction(self.snapshots, k)

        #   Generate NearestNeighbors object of data
        if isinstance(self.hf_fit, manifold.SpectralEmbedding):
            self.hf_data_nbrs = NearestNeighbors(n_neighbors=k)
            self.hf_data_nbrs.fit(self.snapshots)
        else:
            self.hf_data_nbrs = self.hf_fit.nbrs_

        #   Extract high fidelity embedding 
        if self.scale_embedding:
            self.embedding_scaler = MinMaxScaler()
            self.embedding_scaler.fit(self.hf_fit.embedding_)
            self.embedding = \
                    self.embedding_scaler.transform(self.hf_fit.embedding_)
        else:
            self.embedding = self.hf_fit.embedding_

        #   Fit Nearest Neighbors in embedding space
        self.embedding_nbrs = NearestNeighbors(n_neighbors=k)
        self.embedding_nbrs.fit(self.embedding)

        # Fit manifold for each low fidelity set
        for i in range(len(self.linked_lf_snapshots)):
            combined_data = np.vstack(
                                (self.linked_lf_snapshots[i], 
                                 self.unlinked_lf_snapshots[i])) 

            # Extract value of k
            if type(self.k_neighbors) is int:
                k = self.k_neighbors
            else:
                k = self.k_neighbors[i+1]

            # Perform reduction and store
            lf_fit = self._fit_single_reduction(combined_data, k)
            self.lf_fits.append(lf_fit)

            # Generate NearestNeighbors object of data
            if isinstance(self.hf_fit, manifold.SpectralEmbedding):
                data_nbrs = NearestNeighbors(n_neighbors=k)
                data_nbrs.fit(combined_data)
                self.lf_data_nbrs.append(data_nbrs)

            else:
                self.lf_data_nbrs = lf_fit.nbrs_

            # Extract embedding 
            if self.scale_embedding:
                scaler = MinMaxScaler()
                scaler.fit(lf_fit.embedding_)
                self.lf_embedding_scalers.append(scaler)
                self.lf_embeddings.append(scaler.transform(lf_fit.embedding_))
            else:
                self.lf_embeddings.append(lf_fit.embedding_) 

            # Fit Nearest Neighbors in embedding space
            embedding_nbrs = NearestNeighbors(n_neighbors=k)
            embedding_nbrs.fit(self.lf_embeddings[i])
            self.lf_embedding_nbrs.append(embedding_nbrs)

        # Only store the value of k for the high fidelity data for backmapping
        if type(self.k_neighbors) is tuple:
            self.k_neighbors = self.k_neighbors[0]

    def _compute_optimal_k(self, metric='best'):
        ''' Implement algorithm for computing the optimal value of k by
            minimizing Kurskal's stress or the variance of distance
            ratios

        --- Inputs ---
        metric : str (default 'best')
            Objective function to use when optimizing k. If best, use 
            Kruskal's stress for Isomap and variance of distance ratios
            otherwise ['best', 'kruskal', 'sammon', 'resid', 'variance'] 
        '''
        # Initialize list of possible k values 
        if self.method == 'hessian' or self.method == 'hess':
            k_list = list(range(self.n_components \
                                + n_components*(n_components+1)//2,
                                self.num_snapshots-1))
        else:
            k_list = list(range(self.n_components+1, self.num_snapshots-1))

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
                                    w_bounds=(-0.25, 1.25), max_iter=10):
        ''' Check if reconstruction points lie within bounds at interp points
            If not, increase k until all weights lie within bounds or until
            max k is reached
        
        --- Inputs ---
        validation_params : (n_predictions, n_params) numpy.ndarray
            Array of points at which predicitons are made
        w_bounds : tuple (default (-0.25, 1.25))
            Min and max values of reconstruction weights for
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
