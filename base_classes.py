''' Base classes for ROM objects

Written by Kenneth Decker
kdecker8@gatech.edu
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
import dill

from scipy.spatial import Delaunay
from scipy.optimize import minimize, basinhopping, differential_evolution
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, decomposition
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import pairwise_distances
# from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.utils.graph import single_source_shortest_path_length as graph_shortest_path
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as gpk
from sklearn.neural_network import MLPRegressor
from scipy import interpolate
# from tensorflow import keras

from hierarchichal_kriging import *
#from multifidelity_rbf import *

from pyToolBox.surrogate import KrigingBuilder, MFKrigingBuilder


class BaseClass(object):
    ''' Base class for all ROMs

    --- Attributes ---

    --- Methods ---

    '''

    def edit_settings(self, **kwargs):
        ''' Edit settings after after object has already been 
            instantiated. Keyword arguments correspond to settings 
            listed in __init__
        '''
        keys = kwargs.keys()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError('Attribute <%s> not found' % key)

    def read_training_snapshots(self, snapshots):
        ''' Read in snapshots to train

        --- Inputs ---
        snapshots : (n_snapshots, n_grid) numpy.ndarray
            Array of training snapshots
        '''
        n_snapshots, n_grid = np.shape(snapshots)
        if hasattr(self, 'training_params'):
            if n_snapshots != self.training_params.shape[0]:
                raise ValueError('Number of snapshots must match number\
                    of training parameter points')

        self.num_snapshots = n_snapshots
        self.grid_dimension = n_grid

        if self.scale_outputs:
            self.outputScaler = MinMaxScaler()
            self.outputScaler.fit(snapshots)
            self.snapshots = self.outputScaler.transform(snapshots)

        else:
            self.outputScaler = None
            self.snapshots = snapshots

    def read_training_params(self, training_params, param_ranges=None):
        ''' Read design parameters corresponding to training snapshots

        --- Inputs ---
        training_params : (n_snapshots, n_params) numpy.ndarray
            Array of training parameters corresponding to training snapshots
        param_ranges : tuple (default None)
            Tuple of tuple pairs where the entries of the ith tuple 
            correspond to min and max values of the ith component. If None, 
            defaults to min and max values provided in training_params
        '''
        if hasattr(self, 'snapshots'):
            if training_params.shape[0] != self.num_snapshots:
                raise ValueError('Number of training parameter points must\
                    match number of training snapshots')

        self.n_training_params = training_params.shape[1]
        if self.scale_inputs:
            self.inputScaler = MinMaxScaler()
            self.inputScaler.fit(training_params)
            self.training_params = self.inputScaler.transform(training_params)
            self.param_ranges = ((0,1),)*training_params.shape[1]

        else:
            self.inputScaler = None
            self.training_params = training_params

            if param_ranges is None:
                self.param_ranges = tuple(
                    [(training_params.min(axis=0)[i], 
                        training_params.max(axis=0)[i]) \
                        for i in range(training_params.shape[1])]
                )
            else:
                self.param_ranges = param_ranges

    def save_model(self, filepath):
        ''' Save model using dill or keras save method 

        --- Inputs ---
        filepath: str
            file path to save model
        ''' 
        if hasattr(self, 'encoder'):
            self.encoder.save(filepath + '_encoder', save_format='tf')
            self.decoder.save(filepath + '_decoder', save_format='tf')

        else:
            fileID = open(filepath, 'wb')
            dill.dump(self, fileID)

    def _fit_interpolation(self):
        ''' Generate interpolation models for embedding coefficients
        '''
        def GA(func, theta0, bounds):
            opt = differential_evolution(func, args=(False,),
                                    bounds=bounds, tol=1e-10, popsize=30)
            return opt.x, opt.fun

        # Generate interpolation function and append to list
        if self.interp_method == 'rbf':
            if self.interp_kernel in ['multiquadric', 'cubic', 'gaussian', \
            'inverse_multiquadric', 'linear', 'quintic', 'thin_plate']:

                interp_function = interpolate.Rbf(
                    *self.training_params.T, 
                    self.embedding, 
                    function=self.interp_kernel,
                    epsilon=self.interp_parameter,
                    mode='N-D')
                self._interp_function = lambda X: interp_function(*X.T)

            else:
                raise RuntimeError('Inteprolation kerenel not supported, \
                    must be ["multiquadric", "cubic", \
                    "gaussian","inverse_multiquadric", "linear", "quintic", \
                    "thin_plate"]')               

        elif self.interp_method == 'gauss':
            length_scale = np.mean(self.embedding.max(axis=0) 
                - self.embedding.min(axis=0))/float(self.num_snapshots)

            if self.interp_kernel == 'constant':
                if self.interp_parameter == None:
                    kernel = gpk.ConstantKernel(
                        constant_value=self.embedding.mean(axis=None))
                else:
                    kernel = gpk.ConstantKernel(
                        constant_value=self.interp_parameter)

            elif self.interp_kernel == 'sine':
                if self.interp_parameter == None:
                    period = np.max(self.embedding.max(axis=0) 
                        - self.embedding.min(axis=0))
                    kernel = gpk.ExpSineSquared(length_scale=length_scale, 
                                                periodicity=period)
                else:
                    kernel = gpk.ExpSineSquared(**self.interp_parameter)

            elif self.interp_kernel == 'dot':
                if self.interp_parameter == None:
                    kernel = gpk.DotProduct(sigma_0=0.0)
                else:
                    kernel = gpk.DotProduct(sigma_0=self.interp_parameter)

            elif self.interp_kernel == 'rbf':
                if self.interp_parameter == None:
                    kernel = gpk.RBF(length_scale=length_scale)
                else:
                    kernel = gpk.RBF(length_scale=self.interp_parameter)

            elif self.interp_kernel == 'matern' or self.interp_kernel is None:
                if self.interp_parameter == None:
                    kernel = gpk.Matern(length_scale=length_scale, nu=1.5)
                else:
                    kernel = gpk.Matern(**self.interp_parameter)

            elif self.interp_kernel == 'quad':
                if self.interp_parameter == None:
                    kernel = gpk.RationalQuadratic(
                                    length_scale=length_scale, alpha=1.0)
                else:
                    kernel = gpk.RationalQuadratic(**self.interp_parameter)

            elif isinstance(self.interp_kernel, gpk.Kernel):
                raise RuntimeError('Arbitrary Kernel not yet supported')
                if self.interp_parameter == None:
                    pass
                else:
                    pass

            else:
                raise RuntimeError('Inteprolation kerenel not supported \
                        must be "constant", "sine", "dot", "rbf",\
                        "matern", "quad"] or isntance of \
                        sklearn.gaussian_process.kernels.Kernel')
            
            self.gp_regressor = GaussianProcessRegressor(kernel=kernel,
                                    alpha=1e-8, optimizer=GA, normalize_y=True)

            self.gp_regressor.fit(self.training_params, self.embedding)
            self._interp_function = self.gp_regressor.predict 

        elif self.interp_method == 'gauss-t':
            krig_builder = KrigingBuilder(kernel=self.interp_kernel,
                                          trend='linear', optimize=True,
                                          optim_restarts=5, regularize=True)
            self.gp_regressor = krig_builder.train(self.training_params, 
                                                   self.embedding)
            self._interp_function = self.gp_regressor.eval

        elif self.interp_method == 'nn':
            self.nn_regressor = MLPRegressor(**self.interp_parameter)

            if self.scale_inputs and self.scale_embedding:
                self.nn_regressor.fit(self.training_params, self.embedding)
                self._interp_function = self.nn_regressor.predict

            elif self.scale_inputs:
                self.nn_embedding_scaler = MinMaxScaler()
                self.nn_embedding_scaler.fit(self.embedding)
                self.nn_regressor.fit(self.training_params, 
                        self.nn_embedding_scaler.transform(self.embedding))

                if self.n_components == 1:
                    self._interp_function = lambda X: \
                            self.nn_embedding_scaler.inverse_transform(
                                    np.reshape(self.nn_regressor.predict(X), 
                                        (1,-1)))
                else:
                    self._interp_function = lambda X: \
                            self.nn_embedding_scaler.inverse_transform(
                                    self.nn_regressor.predict(X))

            elif self.scale_embedding:
                self.nn_inputs_scaler = MinMaxScaler()
                self.nn_inputs_scaler.fit(self.training_params)
                self.nn_regressor.fit(
                    self.nn_inputs_scaler.transform(self.training_params),
                                                    self.embedding)

                self._interp_function = lambda X: \
                        self.nn_regressor.predict(
                            self.nn_inputs_scaler.transform(X))

            else:
                self.nn_inputs_scaler = MinMaxScaler()
                self.nn_inputs_scaler.fit(self.training_params)
                self.nn_embedding_scaler = MinMaxScaler()
                self.nn_embedding_scaler.fit(self.embedding)

                self.nn_regressor.fit(
                    self.nn_inputs_scaler.transform(self.training_params),
                    self.nn_embedding_scaler.transform(self.embedding))

                if self.n_components == 1:
                    self._interp_function = lambda X: \
                        self.nn_embedding_scaler.inverse_transform(
                            np.reshape(self.nn_regressor.predict(
                                self.nn_inputs_scaler.transform(X)), (1,-1)))
                else:
                    self._interp_function = lambda X: \
                        self.nn_embedding_scaler.inverse_transform(
                            self.nn_regressor.predict(
                                self.nn_inputs_scaler.transform(X)))

        elif self.interp_method == 'linear':
            interp_function = interpolate.LinearNDInterpolator(
                                        self.training_params, self.embedding)
            self._interp_function = interp_function

        elif self.interp_method == 'rbf-local':
            self.rbf_regressor = SingleFidelityLocalRBF(
                kernel=self.interp_kernel,
                param=self.interp_parameter,
                optimize=True,
                optimizer='L-BFGS-B',
                order=None,
                normalize=False,
                alpha=1e-10,
                n_restarts=10,
                k_neighbors=30,
                leafsize=100,
                n_jobs=self.n_jobs
            )
            self.rbf_regressor.fit(self.training_params, self.embedding)
            self._interp_function = self.rbf_regressor.predict

        else:
            print(self.interp_method)
            raise RuntimeError('Interpolation method not supported, \
                    must be ["rbf", "gauss", "nn"]')

    def _compute_reduction_diagnostics(self):
        ''' Compute Kruskal stress, Sammon stress, residual variance, and 
            variance of distance ratios of low-dimensional embedding
        '''
        # Extract geodesic distances for Local methods
        if isinstance(self.fit, manifold.LocallyLinearEmbedding) \
        or isinstance(self.fit, manifold.SpectralEmbedding):

            params = self.fit.get_params()

            kng = kneighbors_graph(
                    self.data_nbrs, params['n_neighbors'], 
                    metric='minkowski', p=2, metric_params=None, 
                    mode='distance', n_jobs=params['n_jobs'])

            d = graph_shortest_path(kng, method='auto', directed=False)

        elif isinstance(self.fit, manifold.Isomap):
            d = self.fit.dist_matrix_

        # Supress divide by zero warning for diagnostic computations
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # Construct necessary distance matrices
        delta = pairwise_distances(self.embedding)

        d_sq = np.square(d)
        delta_sq = np.square(delta)

        diff_matrix = d - delta
        diff_sq = np.square(diff_matrix)

        # Compute Kruskal and Sammon stress
        self.k_stress = np.sqrt(np.sum(np.square(diff_matrix), axis=None)\
            /np.sum(d_sq, axis=None))
        self.s_stress = np.sqrt(np.sum(np.square(diff_sq), axis=None)\
            /np.sum(np.square(d_sq), axis=None))

        # Compute residual variance
        rho = np.corrcoef(np.reshape(d, (1,-1)), np.reshape(delta, (1,-1)))
        self.residual_variance = 1.0 - (rho[0,1]**2.)

        # Compute variance of distance ratios
        ratio = np.divide(np.reshape(delta, (1,-1)), np.reshape(d, (1,-1)))
        ratio = ratio[~np.isnan(ratio)]
        self.ratio_variance = np.var(ratio)

        # Turn divide by zero warnings back on
        warnings.filterwarnings('default', category=RuntimeWarning)

    def _check_extrapolation(self, prediction_params):
        ''' Check if prediction points are extrapolated based on a Delaunay 
            triangulation of the embedding space
        '''
        pass


class BaseManifoldBackmap(object):
    ''' Base class for locally linear manifold backmapping

    --- Attributes ---

    --- Methods ---

    '''

    def _generate_gram_matrix(self, y_star):
        ''' Generate Gram matrix, used for backmapping

        --- Inputs ---
        y_star : (n_dim, 1) numpy.ndarray
            Point to evaluated backmapping, provided in manifold coordinates
        '''
        # Get k nearest neighbors of y_star and their respective distances
        dist, inds = self.embedding_nbrs.kneighbors(y_star.T)

        # Generate matrix of vectors from y_star to each neighbor
        R = self.embedding[inds[0],:].T
        R_shift = y_star - R

        # Compute Gram matrix of neighbor vectors
        G = np.matmul(R_shift.T, R_shift)

        return (G, R_shift, inds)

    def _solve_mapping_weights(self, y_star):
        ''' Solve for the optimal weights that are used to reconstruct data at 
            prediction point

        --- Inputs ---
        y_star : (n_dim, 1) numpy.ndarray
            Point to evaluated backmapping, provided in manifold coordinates
        '''
        G, R, inds = self._generate_gram_matrix(y_star)
        dim, samples = np.shape(R)

        # Generate array of costs for regularization
        norm_vec = [la.norm(R[:, i]) for i in range(samples)]
        c_vec = [self.eps*((norm_vec[i]/np.max(norm_vec))**self.p) \
                                            for i in range(len(norm_vec))]
        
        # Form matrices used to perform optimization 
        C = np.diag(c_vec)
        G_bar = G + C

        zero_vec = np.zeros((samples, 1))
        one_vec = np.ones((samples, 1))

        A = np.vstack((np.hstack((2*G_bar, one_vec)), \
                np.reshape(np.append(one_vec.T, 0), (1,-1))))
        b = np.vstack((zero_vec, 1))
        
        # Solve for optimal weights
        soln_vec = la.solve(A, b)
        w_vec = soln_vec[0:samples]
        lam = soln_vec[-1]

        return (w_vec, lam, inds)

    def _compute_condition_number(self, y_star):
        ''' Build matrix for computing backmapping wieghts and comput condition 
        number

        --- Inputs ---
        y_star : (n_dim, 1) numpy.ndarray
            Point to evaluated backmapping, provided in manifold coordinates

        --- Returns ---
        cond : float
            Condition number of backmapping matrix
        '''
        # Generate Gram Matrix
        G, R, inds = self._generate_gram_matrix(y_star)
        dim, samples = np.shape(R)

        # Generate array of costs for regularization
        norm_vec = [la.norm(R[:, i]) for i in range(samples)]
        c_vec = [self.eps*((norm_vec[i]/np.max(norm_vec))**self.p) \
                                            for i in range(len(norm_vec))]
        
        # Form matrices used to perform optimization 
        C = np.diag(c_vec)
        G_bar = G + C

        zero_vec = np.zeros((samples, 1))
        one_vec = np.ones((samples, 1))

        A = np.vstack((np.hstack((2*G_bar, one_vec)), \
                np.reshape(np.append(one_vec.T, 0), (1,-1))))

        return la.cond(A)

    def _compute_manifold_backmap(self, y_star, return_weights=False):
        ''' Solve for backmapping wieghts and perform reconstruction

        --- Inputs ---
        y_star : (1, n_dim) numpy.ndarray
            Point to evaluated backmapping, provided in manifold coordinates
        return_weights : bool (default False)
            If true, return weight vector along with result
        '''
        w_vec, lam, inds = self._solve_mapping_weights(y_star.T)

        full_vecs = self.snapshots[inds[0],:].T

        contribution = full_vecs*w_vec.T

        if return_weights:
            return (w_vec.flatten(), np.sum(contribution, axis=1))
        else:
            return np.sum(contribution, axis=1)


class BaseAdaptiveSampling(object):
    ''' Base class for adaptive sampling techniques

    --- Attributes ---

    --- Methods ---

    '''
    def perform_adaptive_sampling(self, metric='MDE', n=1, **kwargs):
        ''' Identify the next set of design parameters to analyze when 
            performing adaptive sampling

        --- Inputs ---
        metric : str (default 'MDE')
            Adaptive sampling metric to use when making prediction
            ['mde', 'var']
        n : int (default 1)
            Number of adaptive sampling points to generate

        --- Returns ---
        adapt_points: (n, n_dvs) numpy.ndarray
            Array of points to use for sequential sampling
        '''
        # Set up dictionaries for optimizer settings using kwargs
        restarts = kwargs.get('restarts', 10)
        T = np.mean(self.embedding.max(axis=0) 
            - self.embedding.min(axis=0))/float(self.num_snapshots)
        mins = np.array([i[0] for i in self.param_ranges])
        maxs = np.array([i[1] for i in self.param_ranges])
        max_step = min((maxs-mins)/2.)

        local_settings = {
            'method': kwargs.get('optimizer', 'SLSQP'),
            'bounds': self.param_ranges,
            'tol': kwargs.get('tol', 1e-8),
            'options': {
                'maxiter': kwargs.get('maxiter_local', 100), 
                'disp': False
            },
        }

        global_settings = {
            'niter': kwargs.get('niter', 100),
            'T': T,
            'stepsize': max_step,
            'disp': False,
            'niter_success': kwargs.get('niter_success', 10),
            'minimizer_kwargs': local_settings,
        }

        # Carry out multiple iterations of basinhopping optimization
        count = 0
        sample_points = np.zeros((n, self.training_params.shape[1]))
        while True:
            # Define local function for maximum kriging variance objective
            def estimator_variance(X):
                X = np.atleast_2d(X)
                m, s = self._interp_function(X, return_std=True)
                if type(s) is float:
                    return -s

                return -s.mean()

            # Define local function for hybrid variance metric
            def hybrid_variance(X, p_nbrs):
                X = np.atleast_2d(X)
                m, s = self._interp_function(X, return_std=True)
                E_p = self._compute_dv_dist(X, p_nbrs)

                return E_p*s.mean() # E_p is already negative

            # Define local function for wieght norm vector
            def weight_norm(X):
                X = np.atleast_2d(X)
                _, w = self.predict(X, return_weights=True)

                w_norms = la.norm(w-0.5, axis=1)

                return -w.sum()

            # set up temporary neighbors calculation
            if metric.lower() in ['mde', 'rec', 'hyb-dist']:
                if count > 0:
                    tmp_nbrs = NearestNeighbors(n_neighbors=self.k_neighbors)
                    tmp_nbrs.fit(self.embedding)
                else:
                    tmp_nbrs = self.embedding_nbrs

            if metric.lower() in ['hyb-dist', 'hyb-var']:
                p_nbrs = NearestNeighbors(n_neighbors=self.k_neighbors)
                p_nbrs.fit(self.training_params)

            # Set up temporary interpolator if variance calculation needed
            if metric.lower() in ['var', 'hyb-var']:
                if self.interp_method not in ['gauss']:
                    original_interp_method = self.interp_method
                    original_interp_kernel = self.interp_kernel
                    original_interp_param = self.interp_parameter
                    self.edit_settings(interp_method='gauss',
                                        interp_kernel='matern',
                                        interp_parameter=None)
                    self._fit_interpolation()
                    switchFlag = True
                else:
                    switchFlag = False

            # Set up objective
            if metric.lower() == 'mde':
                local_settings['args'] = (tmp_nbrs)
                obj = self._compute_y_dist

            elif metric.lower() == 'rec':
                predictions = self.predict(self.training_params)
                E_rel = la.norm(predictions-self.snapshots, axis=1)
                E_rel /= la.norm(self.snapshots, axis=1)

                krig_builder = KrigingBuilder(kernel='matern32', 
                                                trend='linear',
                                                optimize=True,
                                                optim_restarts=5,
                                                regularize=True)
                error_krig = krig_builder.train(self.embedding, E_rel)
                local_settings['args'] = (error_krig.eval,tmp_nbrs)
                obj = self._compute_reconstruction_metric

            elif metric.lower() == 'var':
                obj = estimator_variance

            elif metric.lower() == 'hyb-dist':
                local_settings['args'] = (p_nbrs, tmp_nbrs)
                obj = self._compute_hybrid_dist_metric

            elif metric.lower() == 'hyb-var':
                local_settings['args'] = (p_nbrs)
                obj = hybrid_variance

            elif metric.lower() == 'w-norm':
                obj = weight_norm

            # Carry out sequential sampling by optimizing objective
            E_opt = np.inf
            p_opt = None
            for i in range(restarts):
                p_test = np.random.rand(1, self.n_training_params)
                p_test = mins + (maxs-mins)*p_test

                try:
                    opt = basinhopping(obj, x0=p_test, **global_settings)
                except ValueError:
                    continue

                if 'success condition satisfied' in opt.message:
                    if opt.fun < E_opt:
                        E_opt = opt.fun
                        p_opt = opt.x

            if self.scale_inputs:
                sample_points[count,:] = self.inputScaler.inverse_transform(p_opt)
            else:
                sample_points[count,:] = p_opt

            # If n simulations have been carried out, terminate run
            count += 1
            if count == n:
                s = tuple(range(-n+1,0))
                s = [i+self.training_params.shape[0] for i in s] 
                self.embedding = np.delete(self.embedding, s, axis=0)
                self.training_params = np.delete(self.training_params, s, axis=0)
                self.snapshots = np.delete(self.snapshots, s, axis=0)
                break

            tmp_embedding = np.atleast_2d(
                self._interp_function(np.atleast_2d(p_opt))
            )
            self.embedding = np.vstack((self.embedding, tmp_embedding))
            self.training_params = np.vstack((self.training_params, p_opt))
            self.snapshots = np.vstack((self.snapshots, 
                                        self.predict(np.atleast_2d(p_opt))))

            if metric.lower() == 'var':
                self._fit_interpolation()

        if metric.lower() == 'var' and switchFlag:
            self.edit_settings(
                interp_method=original_interp_method,
                interp_kernel=original_interp_kernel,
                interp_parameter=original_interp_param
            )
            self._fit_interpolation()

        return sample_points

    def _compute_reconstruction_metric(self, p_test, interp_func, nbrs):
        ''' Compute reconstruction metric at an out of sample point

        --- Inputs ---
        p_test: (n_dv,) numpy.ndarray
            Combination of DVs to test
        interp_func: callable
            Function used to estimate relative error at out of sample
            point
        nbrs: sklearn.neighbors.NearestNeighbors
            Scikit-learn nearest neighbors object

        --- Returns ---
        E_rec:
            Reconstruction error at test point 
        '''
        E_dist = self._compute_y_dist(p_test, nbrs)
        y = np.atleast_2d(self._interp_function(p_test))
        E_rel = interp_func(y)
        return -E_rel*E_dist

    def _compute_y_dist(self, p_test, nbrs):
        ''' Compute sum of distances between a test design point and its
            k nearest neighbors after being interpolated to embedding space

        --- Inputs --- 
        p_test: (1, n_components) numpy.ndarray
            Array of design variables to test
        nbrs: sklearn.neighbors.NearestNeighbors
            Scikit-learn nearest neighbors object fit to embedding points

        --- Returns ---
        E_dist: float
            Negative sum of distances
        '''
        p_test = np.atleast_2d(p_test)
        y_test = np.atleast_2d(self._interp_function(p_test))
        dist, inds = nbrs.kneighbors(y_test, return_distance=True)
        d_min = dist.min(axis=None)
        d_max = dist.max(axis=None)

        E_dist = (d_min/d_max)*np.sum(dist, axis=None)

        return -E_dist

    def _compute_dv_dist(self, p_test, nbrs):
        ''' Compute sum of distances between a test design point and its
            k nearest neighbors after being interpolated to embedding space

        --- Inputs --- 
        p_test: (1, n_components) numpy.ndarray
            Array of design variables to test
        nbrs: sklearn.neighbors.NearestNeighbors
            Scikit-learn nearest neighbors object fit to training points

        --- Returns ---
        E_dist: float
            Negative sum of distances
        '''
        p_test = np.atleast_2d(p_test)
        dist, inds = nbrs.kneighbors(p_test, return_distance=True)
        d_min = dist.min(axis=None)
        d_max = dist.max(axis=None)

        E_dist = (d_min/d_max)*np.sum(dist, axis=None)

        return -E_dist

    def _compute_hybrid_dist_metric(self, p_test, p_nbrs, y_nbrs):
        ''' Compute sum of distances between a test design point and its
            k nearest neighbors after being interpolated to embedding space

        --- Inputs --- 
        p_test: (1, n_components) numpy.ndarray
            Array of design variables to test
        nbrs: sklearn.neighbors.NearestNeighbors
            Scikit-learn nearest neighbors object

        --- Returns ---
        E_dist: float
            Negative sum of distances
        '''
        E_y = self._compute_y_dist(p_test, y_nbrs)
        E_p = self._compute_dv_dist(p_test, p_nbrs)

        return -E_y*E_p


class BaseAlignment(object):
    ''' Base class for multifidelity ROMs using manifold alignment

    --- Inherits ---

    --- Attributes ---

    --- Methods ---

    '''
    def edit_settings(self, **kwargs):
        ''' Edit settings after after object has already been 
            instantiated. Keyword arguments correspond to settings 
            listed in __init__
        '''
        keys = kwargs.keys()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError('Attribute <%s> not found' % key)

    def read_high_fidelity_snapshots(self, snapshots):
        ''' Read in snapshots to train

        --- Inputs ---
        snapshots : (n_snapshots, n_grid) numpy.ndarray
            Array of training snapshots
        '''
        n_snapshots, n_grid = np.shape(snapshots)
        if hasattr(self, 'params'):
            if n_snapshots != self.params.shape[0]:
                raise ValueError('Number of snapshots must match number\
                    of training parameter points')

        self.num_snapshots = n_snapshots
        self.grid_dimension = n_grid

        if self.scale_outputs:
            self.outputScaler = MinMaxScaler()
            self.outputScaler.fit(snapshots)
            self.snapshots = self.outputScaler.transform(snapshots)

        else:
            self.outputScaler = None
            self.snapshots = snapshots

    def read_high_fidelity_params(self, training_params, param_ranges=None):
        ''' Read design parameters corresponding to training snapshots

        --- Inputs ---
        training_params : (n_snapshots, n_params) numpy.ndarray
            Array of training parameters corresponding to training snapshots
        param_ranges : tuple (default None)
            Tuple of tuple pairs where the entries of the ith tuple 
            correspond to min and max values of the ith component. If None, 
            defaults to min and max values provided in training_params
        '''
        if hasattr(self, 'snapshots'):
            if training_params.shape[0] != self.num_snapshots:
                raise ValueError('Number of training parameter points must\
                    match number of training snapshots')

        self.n_training_params = training_params.shape[1]
        if self.scale_inputs:
            self.inputScaler = MinMaxScaler()
            self.inputScaler.fit(training_params)
            self.params = self.inputScaler.transform(training_params)
            self.param_ranges = ((0,1),)*training_params.shape[1]

        else:
            self.inputScaler = None
            self.params = training_params

            if param_ranges is None:
                self.param_ranges = tuple(
                    [(training_params.min(axis=0)[i], 
                        training_params.max(axis=0)[i]) \
                        for i in range(training_params.shape[1])])

            else:
                self.param_ranges = param_ranges

    def read_low_fidelity_snapshots(self, linked, unlinked):
        ''' Read in low fidelity data to supplement multifidelity analysis

        --- Inputs ---
        linked : (n_linked_snapshots, n_lf_grid) numpy.ndarray
            Array of linked low-fidelity training points
        unlinked : (n_unlinked_snapshots, n_lf_grid) numpy.ndarray
            Array of unlinked low-fidelity training points
        '''
        n_linked, n_lf_grid = linked.shape
        n_unlinked, n_lf_grid = unlinked.shape
        if hasattr(self, 'unlinked_lf_params'):
            if hasattr(self, 'linked_lf_snapshots'):
                data_set_index = len(self.linked_lf_snapshots)
            else:
                data_set_index = 0

            if n_linked != self.num_snapshots:
                raise ValueError('Number of linked low fidelity snapshots \
                                  must match the number of high fidelity \
                                  snapshots')

            if n_unlinked != self.unlinked_lf_params[data_set_index].shape[0]:
                raise ValueError('Number of snapshots in set \
                    %d must match number of params in \
                    set %d' % (data_set_index, data_set_index))

        if self.scale_outputs:
            linked_lf_snapshots = self.outputScaler.transform(linked)
            unlinked_lf_snapshots = self.outputScaler.transform(unlinked)
        else:
            linked_lf_snapshots = linked
            unlinked_lf_snapshots = unlinked
        
        if hasattr(self, 'linked_lf_snapshots'):
            self.linked_lf_snapshots.append(linked_lf_snapshots)
            self.unlinked_lf_snapshots.append(unlinked_lf_snapshots)
        else:
            self.linked_lf_snapshots = [linked_lf_snapshots]
            self.unlinked_lf_snapshots = [unlinked_lf_snapshots]

    def read_low_fidelity_params(self, unlinked_params):
        ''' Read in low fidelity data to supplement multifidelity analysis

        --- Inputs ---
        unlinked_params : (n_unlinked_params, n_grid) numpy.ndarray
            Array of training snapshots
        '''
        n_unlinked, n_lf_grid = unlinked_params.shape
        if hasattr(self, 'unlinked_lf_snapshots'):
            if hasattr(self, 'unlinked_lf_params'): 
                data_set_index = len(self.lf_params)
            else:
                data_set_index = 0
                
            if n_unlinked \
            != self.unlinked_lf_snapshots[data_set_index].shape[0]:
                raise ValueError('Number of params in set \
                    %d must match number of snapshots in \
                    set %d' % (data_set_index, data_set_index))

        if self.scale_inputs:
            unlinked_lf_params = self.inputScaler.transform(unlinked_params)
        
        if hasattr(self, 'unlinked_lf_params'):
            self.unlinked_lf_params.append(unlinked_params)
        else:
            self.unlinked_lf_params = [unlinked_params]

    def save_model(self, filepath):
        ''' Save model using dill 

        --- Inputs ---
        filepath: str
            file path to save model
        ''' 
        fileID = open(filepath, 'wb')
        dill.dump(self, fileID)

    def _align_manifolds(self):
        ''' Use Procrustes analysis to align low-fidleity manifolds to 
            the high fidelity manifold
        '''
        # Initialize scaler to center manifold data and center hf data
        centerer = StandardScaler(with_std=False)
        centerer.fit(self.embedding)
        centered_hf_embedding = centerer.transform(self.embedding)
        self.hf_embedding = centered_hf_embedding 
        
        # Loop through linked data sets and align using Procrustes 
        for i in range(len(self.lf_embeddings)):
            # Extract lf linked data and center
            linked_lf_embedding = \
                            self.lf_embeddings[i][:self.num_snapshots,:]
            centerer.fit(linked_lf_embedding)
            centered_lf_embedding = centerer.transform(linked_lf_embedding) 

            # Perform SVD
            U, S, Vh = la.svd(np.dot(centered_lf_embedding.T, 
                                     centered_hf_embedding))
            s = S.sum(axis=None)/(la.norm(centered_lf_embedding)**2)
             
            Q = np.dot(U,Vh)
            self.lf_embeddings[i] = s*np.dot(self.lf_embeddings[i],Q) 

        # Combine embeddings and training parameters
        unlinked_embeddings = [x[self.num_snapshots:,:] \
                                            for x in self.lf_embeddings]
        self.embedding_combined = np.vstack([self.embedding] 
                                            + unlinked_embeddings)
        self.params_combined = np.vstack([self.params] 
                                         + self.unlinked_lf_params) 
        
    def _fit_interpolation(self):
        ''' Generate interpolation models for embedding coefficients
        '''
        # Define Genetic Algorithm for fitting Kriging models
        def GA(func, theta0, bounds):
            opt = differential_evoluation(func, args=(False,),
                                bounds=bounds, tol=1e-10, popsize=30)
            return opt.x, opt.fun

        # Generate interpolation function and append to list
        if self.interp_method == 'rbf':
            if self.interp_kernel in ['multiquadric', 'cubic', 'gaussian', \
            'inverse_multiquadric', 'linear', 'quintic', 'thin_plate']:

                interp_function = interpolate.Rbf(
                    *self.params_combined.T, 
                    self.embedding_combined, 
                    function=self.interp_kernel,
                    epsilon=self.interp_parameter,
                    mode='N-D')
                self._interp_function = lambda X: interp_function(*X.T)

            else:
                raise RuntimeError('Inteprolation kerenel not supported, \
                    must be ["multiquadric", "cubic", \
                    "gaussian","inverse_multiquadric", "linear", "quintic", \
                    "thin_plate"]')

        elif self.interp_method == 'gauss':
            def GA(func, theta0, bounds):
                opt = differential_evolution(func, args=(False,),
                                    bounds=bounds, tol=1e-10, popsize=30)
                return opt.x, opt.fun

            length_scale = np.mean(self.embedding_combined.max(axis=0) 
                                 - self.embedding_combined.min(axis=0))\
                            /float(self.num_snapshots)

            if self.interp_kernel == 'constant':
                if self.interp_parameter == None:
                    kernel = gpk.ConstantKernel(
                        constant_value = \
                                self.embedding_combined.mean(axis=None))
                else:
                    kernel = gpk.ConstantKernel(
                        constant_value=self.interp_parameter)

            elif self.interp_kernel == 'sine':
                if self.interp_parameter == None:
                    period = np.max(self.embedding_combined.max(axis=0) 
                        - self.embedding_combined.min(axis=0))
                    kernel = gpk.ExpSineSquared(length_scale=length_scale, 
                                                periodicity=period)
                else:
                    kernel = gpk.ExpSineSquared(**self.interp_parameter)

            elif self.interp_kernel == 'dot':
                if self.interp_parameter == None:
                    kernel = gpk.DotProduct(sigma_0=0.0)
                else:
                    kernel = gpk.DotProduct(sigma_0=self.interp_parameter)

            elif self.interp_kernel == 'rbf':
                if self.interp_parameter == None:
                    kernel = gpk.RBF(length_scale=length_scale)
                else:
                    kernel = gpk.RBF(length_scale=self.interp_parameter)

            elif self.interp_kernel == 'matern':
                if self.interp_parameter == None:
                    kernel = gpk.Matern(length_scale=length_scale, nu=1.5)
                else:
                    kernel = gpk.Matern(**self.interp_parameter)

            elif self.interp_kernel == 'quad':
                if self.interp_parameter == None:
                    kernel = gpk.RationalQuadratic(
                                    length_scale=length_scale, alpha=1.0)
                else:
                    kernel = gpk.RationalQuadratic(**self.interp_parameter)

            elif isinstance(self.interp_kernel, gpk.Kernel):
                raise RuntimeError('Arbitrary Kernel not yet supported')
                if self.interp_parameter == None:
                    pass
                else:
                    pass

            else:
                raise RuntimeError('Inteprolation kerenel not supported \
                        must be "constant", "sine", "dot", "rbf",\
                        "matern", "quad"] or isntance of \
                        sklearn.gaussian_process.kernels.Kernel')
            
            self.gp_regressor = GaussianProcessRegressor(kernel=kernel,
                                    alpha=1e-8, optimizer=GA, normalize_y=True)

            self.gp_regressor.fit(self.params_combined, 
                                  self.embedding_combined)
            self._interp_function = self.gp_regressor.predict

        elif self.interp_method == 'nn':
            self.nn_regressor = MLPRegressor(**self.interp_parameter)

            if self.scale_inputs and self.scale_embedding:
                self.nn_regressor.fit(self.params_combined, 
                                      self.embedding_combined)
                self._interp_function = self.nn_regressor.predict

            elif self.scale_inputs:
                self.nn_embedding_scaler = MinMaxScaler()
                self.nn_embedding_scaler.fit(self.embedding_combined)
                self.nn_regressor.fit(self.params_combined, 
                    self.nn_embedding_scaler.transform(self.embedding_combined))

                if self.n_components == 1:
                    self._interp_function = lambda X: \
                            self.nn_embedding_scaler.inverse_transform(
                                    np.reshape(self.nn_regressor.predict(X), 
                                        (1,-1)))
                else:
                    self._interp_function = lambda X: \
                            self.nn_embedding_scaler.inverse_transform(
                                    self.nn_regressor.predict(X))

            elif self.scale_embedding:
                self.nn_inputs_scaler = MinMaxScaler()
                self.nn_inputs_scaler.fit(self.params_combined)
                self.nn_regressor.fit(
                    self.nn_inputs_scaler.transform(self.params_combined),
                                                    self.embedding_combined)

                self._interp_function = lambda X: \
                        self.nn_regressor.predict(
                            self.nn_inputs_scaler.transform(X))

            else:
                self.nn_inputs_scaler = MinMaxScaler()
                self.nn_inputs_scaler.fit(self.params_combined)
                self.nn_embedding_scaler = MinMaxScaler()
                self.nn_embedding_scaler.fit(self.embedding_combined)

                self.nn_regressor.fit(
                    self.nn_inputs_scaler.transform(self.params_combined),
                    self.nn_embedding_scaler.transform(
                                                self.embedding_combined))

                if self.n_components == 1:
                    self._interp_function = lambda X: \
                        self.nn_embedding_scaler.inverse_transform(
                            np.reshape(self.nn_regressor.predict(
                                self.nn_inputs_scaler.transform(X)), (1,-1)))
                else:
                    self._interp_function = lambda X: \
                        self.nn_embedding_scaler.inverse_transform(
                                    self.nn_regressor.predict(
                                        self.nn_inputs_scaler.transform(X)))

        elif self.interp_method == 'hk':
            length_scale = np.mean(self.embedding.max(axis=0)
                - self.embedding.min(axis=0))/float(self.num_snapshots)

            if self.interp_kernel == 'constant':
                if self.interp_parameter == None:
                    kernel = gpk.ConstantKernel(
                        constant_value=self.embedding.mean(axis=None))
                else:
                    kernel = gpk.ConstantKernel(
                        constant_value=self.interp_parameter)

            elif self.interp_kernel == 'sine':
                if self.interp_parameter == None:
                    period = np.max(self.embedding.max(axis=0) 
                        - self.embedding.min(axis=0))
                    kernel = gpk.ExpSineSquared(length_scale=length_scale, 
                                                periodicity=period)
                else:
                    kernel = gpk.ExpSineSquared(**self.interp_parameter)

            elif self.interp_kernel == 'dot':
                if self.interp_parameter == None:
                    kernel = gpk.DotProduct(sigma_0=0.0)
                else:
                    kernel = gpk.DotProduct(sigma_0=self.interp_parameter)

            elif self.interp_kernel == 'rbf' or self.interp_kernel is None:
                if self.interp_parameter == None:
                    kernel = gpk.RBF(length_scale=length_scale)
                else:
                    kernel = gpk.RBF(length_scale=self.interp_parameter)

            elif self.interp_kernel == 'matern':
                if self.interp_parameter == None:
                    kernel = gpk.Matern(length_scale=length_scale, nu=1.5)
                else:
                    kernel = gpk.Matern(**self.interp_parameter)

            elif self.interp_kernel == 'quad':
                if self.interp_parameter == None:
                    kernel = gpk.RationalQuadratic(
                                    length_scale=length_scale, alpha=1.0)
                else:
                    kernel = gpk.RationalQuadratic(**self.interp_parameter)

            elif isinstance(self.interp_kernel, gpk.Kernel):
                raise RuntimeError('Arbitrary Kernel not yet supported')
                if self.interp_parameter == None:
                    pass
                else:
                    pass

            else:
                raise RuntimeError('Inteprolation kerenel not supported \
                        must be "constant", "sine", "dot", "rbf",\
                        "matern", "quad"] or isntance of \
                        sklearn.gaussian_process.kernels.Kernel')
            
            self.gp_regressor = HierarchichalKriging(kernel=kernel,
                                    alpha=1e-6, optimizer='ga', normalize=True) 
            self.gp_regressor.fit(self.params_combined, self.lf_embeddings[0],
                                                self.params, self.embedding)
            self._interp_function = self.gp_regressor.predict

        elif self.interp_method == 'mfrbf':
            length_scale = np.mean(self.embedding.max(axis=0)
                - self.embedding.min(axis=0))/float(self.num_snapshots)
            
            self.rbf_regressor = MultiFidelityRBF(kernel=self.interp_kernel, 
                                                  alpha=1e-6,
                                                  optimizer='L-BFGS-B', 
                                                  normalize=True, n_restarts=10,
                                                  optimize=True, 
                                                  param=length_scale) 
            self.rbf_regressor.fit(self.params_combined, self.lf_embeddings[0],
                                                self.params, self.embedding)
            self._interp_function = self.rbf_regressor.predict

        elif self.interp_method == 'hk-t':
            krig_builder = MFKrigingBuilder(kernel=self.interp_kernel,
                                            optimize=True, regularize=True,
                                            optim_restarts=5)
            self.gp_regressor = krig_builder.train(self.params, self.embedding,
                                                   self.params_combined,
                                                   self.lf_embeddings[0])
            self._interp_function = self.gp_regressor.eval

        else:
            raise RuntimeError('Interpolation method not supported, \
                    must be ["rbf", "gauss", "nn"]')

    def _compute_reduction_diagnostics(self):
        ''' Compute Kruskal stress, Sammon stress, residual variance, and 
            variance of distance ratios of low-dimensional embedding
        '''
        # Extract geodesic distances for Local methods
        if isinstance(self.hf_fit, manifold.LocallyLinearEmbedding) \
        or isinstance(self.hf_fit, manifold.SpectralEmbedding):

            params = self.hf_fit.get_params()

            kng = kneighbors_graph(
                    self.hf_data_nbrs, params['n_neighbors'], 
                    metric='minkowski', p=2, metric_params=None, 
                    mode='distance', n_jobs=params['n_jobs'])

            d = graph_shortest_path(kng, method='auto', directed=False)

        elif isinstance(self.hf_fit, manifold.Isomap):
            d = self.hf_fit.dist_matrix_

        # Supress divide by zero warning for diagnostic computations
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # Construct necessary distance matrices
        delta = pairwise_distances(self.embedding)

        d_sq = np.square(d)
        delta_sq = np.square(delta)

        diff_matrix = d - delta
        diff_sq = np.square(diff_matrix)

        # Compute Kruskal and Sammon stress
        self.k_stress = np.sqrt(np.sum(np.square(diff_matrix), axis=None)\
            /np.sum(d_sq, axis=None))
        self.s_stress = np.sqrt(np.sum(np.square(diff_sq), axis=None)\
            /np.sum(np.square(d_sq), axis=None))

        # Compute residual variance
        rho = np.corrcoef(np.reshape(d, (1,-1)), np.reshape(delta, (1,-1)))
        self.residual_variance = 1.0 - (rho[0,1]**2.)

        # Compute variance of distance ratios
        ratio = np.divide(np.reshape(delta, (1,-1)), np.reshape(d, (1,-1)))
        ratio = ratio[~np.isnan(ratio)]
        self.ratio_variance = np.var(ratio)

        # Turn divide by zero warnings back on
        warnings.filterwarnings('default', category=RuntimeWarning)

