import numpy as np
import scipy.linalg as la
import sklearn
import sys
import matplotlib.pyplot as plt
import warnings

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as gpk
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.optimize import differential_evolution, minimize


warnings.filterwarnings('ignore')

class HierarchichalKriging():
    ''' Fit hierarchichal kriging regressor to multifidelity data sets

    --- Inputs ---
    kernel: sklearn.gaussian_process.Kernel or callable
        Kernel function to use for regression. If None, defaults to 
        squared exponential kernel (default None)
    alpha: float
        Regularization constant (default 1e-10)
    normalize_y: bool
        normalize y data prior to fitting model (not currently implemented)
    optimize: bool
        Optimize hyperparameter based on maximum log likelihood (default True)
    optimizer: str
        Optimization method. Must be either 'GA' or one of the methods
        accepted by scipy.optimize.minimize (default 'SLSQP')
    n_restarts: None
        Number of times optimizer is restarted, If None, no restarts are 
        performed. If optimizer=='GA', n_restarts is ignored (default None)
    '''
    def __init__(self, kernel=None, alpha=1e-10, normalize=False,
                     optimize=True, optimizer='SLSQP', n_restarts=None):
        self.kernel = kernel
        self.alpha = alpha
        self.normalize = normalize
        self.optimize = optimize
        self.optimizer = optimizer
        self.n_restarts = n_restarts

    def fit(self, x_lf, y_lf, x_hf, y_hf):
        ''' Train Hierarchichal Kriging regressor

        --- Inputs ---
        x_lf: (n_samples_lf, n_features) numpy.ndarray
            Array of low-fidelity inputs
        y_lf: (n_samples_lf, 1) numpy.ndarray
            Array of low-fidleity responses
        x_hf: (n_samples_hf, n_features) numpy.ndarray
            Array of high-fidelity inputs
        y_hf: (n_samples_hf, 1) numpy.ndarray
            Array of high-fidelity responeses
        ''' 

        # Define GA optimizer to find MLLE
        def ga(func, theta0, bounds):
            opt = differential_evolution(func, args=(False,),
                               bounds=bounds, tol=1e-10, popsize=30)
            return opt.x, opt.fun

        def gradient_based(func, theta0, bounds):
            if self.n_restarts is None:
                self.n_restarts = 0

            theta_test = np.linspace(bounds[0,0], bounds[0,1], self.n_restarts)
            theta_test = np.insert(theta_test, 0, theta0)

            opt_theta = []
            opt_obj = []
            for theta in theta_test:
                opt = minimize(func, np.atleast_1d(theta), 
                                args=(False,), method=self.optimizer, 
                                bounds=bounds, tol=1e-8,
                                options={'maxiter': 100, 'disp': False})

                if opt.success:
                    opt_theta.append(opt.x)
                    opt_obj.append(opt.fun)

            theta_star = opt_theta[np.argmin(opt_obj)]
            f_star = np.min(opt_obj)

            return theta_star, f_star
        
        # Define objective function for MLLE optimization
        eps = np.finfo(float).eps
        def obj(theta, arg): 
            self.kernel.theta = theta
            R = pairwise_kernels(X_train, metric=self.kernel)
            R += np.identity(R.shape[0])*self.alpha
            L = la.cholesky(R, lower=True)
            L_inv = la.solve_triangular(L.T, np.identity(L.shape[0]))
            R_inv = np.matmul(L_inv, L_inv.T)
            
            beta = np.matmul(np.matmul(self.F_.T, R_inv), self.F_)
            beta = la.solve(beta, self.F_.T)
            beta = np.matmul(np.matmul(beta, R_inv), y_train)
            
            M = y_train - np.matmul(self.F_,beta)
           
            var = (1./self.n_samples)*(np.matmul(np.matmul(M.T, la.inv(R)), M))
            log_likelihood = -self.n_samples*np.log(var+eps) \
                            - np.log(la.det(R)+eps)

            #return -log_likelihood.item(0) 
            return -np.nansum(log_likelihood)

        # Store training parameters
        if len(y_hf.shape) < 2:
            self.y_train_ = y_hf.reshape((-1,1))
        else:
            self.y_train_ = y_hf

        if len(y_hf.shape) < 2:
            self.y_train_lf_ = y_lf.reshape((-1,1))
        else:
            self.y_train_lf_ = y_lf

        self.X_train_ = np.atleast_2d(x_hf)
        self.X_train_lf_ = np.atleast_2d(x_lf)
        self.n_samples, self.n_features = self.X_train_.shape

        # Set optimizer
        if self.optimizer.lower() == 'ga':
            optimizer = ga
        else:
            optimizer = gradient_based

        # Normalize y values
        self._compute_normalization()
        X_train = self._apply_normalization(
                    self.X_train_, self._X_train_mean, self._X_train_std)
        y_train = self._apply_normalization(
                    self.y_train_, self._y_train_mean, self._y_train_std)
        X_train_lf = self._apply_normalization(
            self.X_train_lf_, self._X_train_mean_lf, self._X_train_std_lf)
        y_train_lf = self._apply_normalization(
            self.y_train_lf_, self._y_train_mean_lf, self._y_train_std_lf)

        # Set default kernel if no kernel is supplied
        if self.kernel is None:
            dist_matrix = pairwise_distances(X_train)
            l = dist_matrix.mean()
            self.kernel = gpk.RBF(length_scale=l)

        # Fit low-fidelity regrssor and evaluate at high-fidelity points
        self.lf_regressor = GaussianProcessRegressor(
                                    kernel=self.kernel,
                                    alpha=self.alpha, 
                                    optimizer=optimizer, 
                                    normalize_y=True
        )
        self.lf_regressor.fit(X_train_lf, y_train_lf)
        self.F_ = self.lf_regressor.predict(X_train)
        
        # Optimize hyperparameters 
        if self.optimize:
            theta_opt, _ = optimizer(obj, self.kernel.theta, self.kernel.bounds)
            self.kernel.theta = theta_opt  

        # Evaluate kernel matrix and store inverse
        R = pairwise_kernels(X_train, metric=self.kernel)
        R += np.identity(R.shape[0])*self.alpha
        L = la.cholesky(R, lower=True)
        L_inv = la.solve_triangular(L.T, np.identity(L.shape[0]))
        self.R_inv_ = np.matmul(L_inv, L_inv.T)

        # Compute beta
        beta_tmp = np.matmul(np.matmul(self.F_.T, self.R_inv_), self.F_)
        beta_tmp = la.solve(beta_tmp, self.F_.T)
        self.beta_ = np.matmul(np.matmul(beta_tmp, self.R_inv_), y_train)

        # Compute and store matrices for prediction 
        M = y_train - np.matmul(self.F_,self.beta_)
        self.V_ = np.matmul(self.R_inv_, M)
        self.p_var_ = (1.0/self.n_samples)*np.matmul(
                                            np.matmul(M.T, self.R_inv_), M)

    def predict(self, X_test, return_std=False):
        ''' Make predictions using multi-fidelity regressor

        --- Inputs ---
        X_test: (n_test_pts, n_features) numpy.ndarray
            Array of test points to evaluate
        return_std: bool
            If True, return prediction std at test points (default False)

        --- Returns ---
        y_mean: (n_test_pts, 1) numpy.ndarray
            Array of mean prediction values at test points
        y_std: (n_test_pts, 1) numpy.ndarray
            Array of std values at test points, returned if 
            return_std == True
        '''
        # Apply normalization
        X_train = self._apply_normalization(
                    self.X_train_, self._X_train_mean, self._X_train_std)
        X_test = self._apply_normalization(
                    X_test, self._X_train_mean, self._X_train_std)

        # Compute mean prediction
        y_lf = self.lf_regressor.predict(X_test)
        R = pairwise_kernels(X_train, X_test, metric=self.kernel)
        
        #y_mean = self.beta_*y_lf + np.matmul(R.T, self.V_)
        y_mean = np.matmul(y_lf,self.beta_) + np.matmul(R.T, self.V_)
        y_mean = self._undo_normalization(
                    y_mean, self._y_train_mean, self._y_train_std)

        if return_std:
            M1 = np.einsum('ij,jk,ki->i', R.T, self.R_inv_, R)
            M2 = np.einsum('ij,jk,ki->i', R.T, self.R_inv_, self.F_) \
                - y_lf.ravel()
            M3 = np.einsum('ij,jk,ki->i', self.F_.T, self.R_inv_, self.F_)

            y_mse = self.p_var_.ravel() * (1. - M1 + M2**2 / M3)
            np.place(y_mse, y_mse < self.alpha, self.alpha)

            y_std = np.sqrt(y_mse)[:, None]
            y_std *= self._y_train_std

            return y_mean, y_std
           
        return y_mean

    def _compute_normalization(self):
        if self.normalize:
            self._X_train_mean = np.mean(self.X_train_, axis=0)
            self._X_train_std = np.std(self.X_train_, axis=0)

            self._y_train_mean = np.mean(self.y_train_, axis=0)
            self._y_train_std = np.std(self.y_train_, axis=0)

            self._X_train_mean_lf = np.mean(self.X_train_lf_, axis=0)
            self._X_train_std_lf = np.std(self.X_train_lf_, axis=0)

            self._y_train_mean_lf = np.mean(self.y_train_lf_, axis=0)
            self._y_train_std_lf = np.std(self.y_train_lf_, axis=0)

        else:
            self._X_train_mean = np.atleast_2d(0.)
            self._X_train_std = 1.
            
            self._X_train_mean_lf = np.atleast_2d(0.)
            self._X_train_std_lf = 1.

            self._y_train_mean = np.atleast_2d(0.)
            self._y_train_std = 1.

            self._y_train_mean_lf = np.atleast_2d(0.)
            self._y_train_std_lf = 1.

    def _apply_normalization(self, x, x_mean, x_std):
        return (x-x_mean)/x_std

    def _undo_normalization(self, x, x_mean, x_std):
        return (x*x_std)+x_mean
