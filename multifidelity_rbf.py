''' Carry out Multifidelity Radial Basis Function Regression '''
import numpy as np 
import scipy.linalg as la 
import sys
import multiprocessing

from scipy.optimize import differential_evolution, minimize
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed


class BaseRBF():
    ''' Base class for carrying out single= and multi-fidelity
        RBF regression
    '''
    def __init__(self, kernel='cubic', param=None, optimize=True,
                order=None, normalize=False, alpha=1e-10,
                optimizer='SLSQP', n_restarts=None, n_jobs=1):
        self.kernel = kernel
        self.param = param
        self.optimize = optimize
        self.order = order
        self.normalize = normalize
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts = n_restarts
        self.n_jobs = n_jobs

    def fit():
        ''' Train regression model '''
        pass

    def predict():
        ''' Evaluate regression model '''
        pass

    def _evaluate_basis(self, X1, X2, kernel, param):
        ''' Evaluate basis function for regression

        --- Inputs ---
        X1: (n1, n_dvs) numpy.ndarray
            Array of inputs to evaluate basis function
        X2: (n2, n_dvs) numpy.ndarray
            Array of inputs to evaluate basis function
        kernel: str
            Name of the basis function for RBF prediction
        param: float
            Hyperparameter for basis function

        --- Returns ---
        Phi: (n1,n2) numpy.ndarray
            Matrix of basis function values
        '''
        m1, n_dvs1 = X1.shape
        m2, n_dvs2 = X2.shape
        if n_dvs1 != n_dvs2:
            raise RuntimeError('Input arrays must have same dimension')

        Phi = np.zeros((m1,m2))
        for i in range(m1):
            for j in range(m2):
                Phi[i,j] = self._basis_function(X1[i,:], X2[j,:], 
                                                kernel, param)
        
        return Phi

    def _evaluate_poly(self, X, order):
        ''' Evaluate polynomial function for regression

        --- Inputs ---
        X: (n_samples, n_dvs) numpy.ndarray
            Array of inputs to evaluate basis function
        order: int
            Order of the polynomial to evaluate
        '''
        m,n = X.shape
        if order == 0:
            return np.ones((m,1))

        elif order == 1:
            return np.hstack((np.ones((m,1)), X))

        # FIX: If higher order is needed, add here

    def _basis_function(self, x1, x2, kernel, param):
        ''' Basis function for two input points

        --- Inputs ---
        x1: (n_dvs,) numpy.ndarray
            First input vector
        x2: (n_dvs,) numpy.ndarray
            Second input vector
        kernel: str
            Name of the basis function for RBF prediction
        param: float
            Hyperparameter for basis function

        --- Returns ---
        phi: float
            Basis function value
        '''
        r = la.norm(x1-x2)
        tiny = 1e-15
        if kernel == 'gaussian':
            exponent = np.sum((np.abs(x1-x2)**2.)*param)
            phi = np.exp(-exponent)
        elif kernel == 'cubic':
            phi = r**3.
        elif kernel == 'thin_plate':
            phi = (r**2.)*np.log(r+tiny)

        # FIX: If other basis functions are needed, add here

        return phi

    def _compute_normalization(self):
        ''' Compute means and stdevs for normalizing inputs and outputs
        '''
        pass

    def _apply_normalization(self, x, x_mean, x_std):
        ''' Normalize vector
        '''
        return (x-x_mean)/x_std

    def _undo_normalization(self, x, x_mean, x_std):
        ''' Undo normalization of a previously normalized vector
        '''
        return (x*x_std)+x_mean


class SingleFidelityRBF(BaseRBF):
    ''' Carry out Radial Basis Regression for single-fielity data

    --- Inputs ---
    kernels: str (default 'cubic')
        Basis function used to for RBF computation, 
        ['cubic', 'thin_plate', 'gaussian']
    param: float (default None)
        Value of parameter in certain interpolation basis functions. 
        If basis function has no parameter, value is ignored. If 
        optimize == True, value is used as initial guess for optimizer.
        If none, value is defaulted.
    optimize: bool (default True)
        If True, optimize hyperparameter using leave-one-out 
        cross-validation.
    optimizer: str (default 'LBFGS-B')
        Type of optimizer to use for hyperpatameter solution. Must
        correspond to options in scipy.minimize
    order: int (default None)
        Order of supplemental polynomial. If -1, no supplemental
        polynomial is used. If None, value is defaulted based on 
        basis function type
    normalize: bool (default True)
        If True, normalize training data prior to fitting model
    alpha: float (default 1e-10)
        Value of ridge regularization parameter
    n_restarts: None (default 10)
        Number of times to restart gradient-based optimizer during 
        model training. If None, no restarts are performed
    '''
    def __init__(self, kernel='cubic', param=None, optimize=True,
                order=None, normalize=True, alpha=1e-10,
                optimizer='SLSQP', n_restarts=None, n_jobs=1):
        super().__init__(kernel=kernel, param=param, optimize=optimize,
                order=order, normalize=normalize, alpha=alpha,
                optimizer=optimizer, n_restarts=n_restarts, n_jobs=n_jobs)

    def fit(self, X, y):
        ''' Train Multifidelity RBF regressor

        --- Inputs ---
        X: (n_samples, n_features) numpy.ndarray
            Array of inputs
        y: (n_samples, n_responses) numpy.ndarray
            Array of low-fidleity responses
        ''' 
        def leave_one_out(param):
            A = self._evaluate_basis(X_train, X_train, self.kernel, param)

            if self.order >= 0:
                F = self._evaluate_poly(X_train, self.order)
                A = np.vstack((np.hstack((A,F)),
                        np.hstack((F.T, np.zeros((F.shape[1], F.shape[1]))))))

            A += np.eye(A.shape[0])*self.alpha
            b = y_train

            if self.order >= 0:
                b = np.vstack((b, np.zeros((F.shape[1], y_train.shape[1]))))

            coeffs = la.solve(A,b)
            betas = coeffs[0:X_train.shape[0],:]

            # Invert A using cholesky decomposition
            L = la.cholesky(A, lower=True)
            L_inv = la.solve_triangular(L.T, np.eye(L.shape[0]))
            A_inv = np.matmul(L_inv, L_inv.T)

            LOO = 0.0
            for i in range(X_train.shape[0]):
                beta = betas[i,:]
                phi_inv = A_inv[i,i]
                LOO += np.sum((beta/phi_inv)**2.)

            return LOO

        # Store training parameters
        if len(y.shape) < 2:
            self.y_train_ = y.reshape((-1,1))
        else:
            self.y_train_ = y

        if len(X.shape) < 2:
            self.X_train_ = X.reshape((-1,1))
        else:
            self.X_train_ = X

        self.n_samples, self.n_features = self.X_train_.shape

        # Normalize y values
        self._compute_normalization()
        X_train = self._apply_normalization(
                    self.X_train_, self._X_train_mean, self._X_train_std)
        y_train = self._apply_normalization(
                    self.y_train_, self._y_train_mean, self._y_train_std)

        # Set default kernel, order and param if either are not provided
        if self.kernel is None:
            self.kernel = 'thin_plate'

        if self.order is None:
            if self.kernel == 'gaussian':
                self.order = -1

            elif self.kernel in ['cubic', 'thin_plate']:
                self.order = 1

            # FIX: If other kernels are provided, set default orders here

        if self.param is None:
            if self.kernel == 'gaussian':
                # self.param = X_train.mean(axis=0)
                self.param = np.ones((X_train.shape[1],))

            else:
                self.param = 1.0

        # Optimize hyperparameter
        if self.optimize and self.kernel in ['gaussian']:
            if self.optimizer is None:
                self.optimizer = 'L-BFGS-B'

            if self.optimizer.lower() == 'ga':
                opt = differential_evolution(
                    func=leave_one_out,
                    bounds=((1e-5,1e5),)*X_train.shape[1],
                    strategy='best1bin',
                    popsize=30,
                    tol=1e-10,
                    mutation=(0.1,0.5),
                    disp=True,
                    polish=True,
                )
                self.param = opt.x

            else:
                if self.n_restarts is None:
                    self.n_restarts = 0

                param_test = np.logspace(-4, 4, self.n_restarts)
                param_test = np.vstack((param_test,)*X_train.shape[1]).T
                param_test = np.vstack((param_test, self.param))

                opt_param = []
                opt_obj = []
                
                for p in param_test:
                    opt = minimize(
                        fun=leave_one_out,
                        x0=p,
                        bounds=((1e-5,1e5),)*X_train.shape[1],
                        method=self.optimizer,
                        tol=1e-10,
                        jac='3-point',
                        options={'maxiter': 100, 'disp': False}
                    )

                    if opt.success:
                        opt_param.append(opt.x)
                        opt_obj.append(opt.fun)

                self.param = opt_param[np.argmin(opt_obj)]

        # Construct training matrices
        A = self._evaluate_basis(X_train, X_train, self.kernel, self.param)

        if self.order >= 0:
            F = self._evaluate_poly(X_train, self.order)
            A = np.vstack((np.hstack((A,F)),
                        np.hstack((F.T, np.zeros((F.shape[1], F.shape[1]))))))
        
        A += np.eye(A.shape[0])*self.alpha
        b = y_train

        if self.order >= 0:
            b = np.vstack((b, np.zeros((F.shape[1], y_train.shape[1]))))

        coeffs = la.solve(A,b)

        # Store alphas and betas
        self.betas_ = coeffs[0:X_train.shape[0],:]

        if self.order >= 0:
            self.alphas_ = coeffs[X_train.shape[0]:,:]

    def predict(self, X_test):
        ''' Evaluate points in X 

        --- Inputs ---
        X_test: (n_test, n_features):
            Array of evaluation points

        --- Return ---
        y_test: (n_test, n_response):
            Array of predictions
        '''
        X_train = self._apply_normalization(
            self.X_train_, self._X_train_mean, self._X_train_std
        )
        X_test = self._apply_normalization(
            X_test, self._X_train_mean, self._X_train_std
        )

        Phi = self._evaluate_basis(
            X_train, X_test, self.kernel, self.param
        )
        y = np.matmul(Phi.T,self.betas_)

        if self.order >= 0:
            Q = self._evaluate_poly(X_test, self.order)
            y += np.matmul(Q, self.alphas_)

        y = self._undo_normalization(y, self._y_train_mean, self._y_train_std)

        return y

    def _compute_normalization(self):
        if self.normalize:
            self._X_train_mean = np.mean(self.X_train_, axis=0)
            self._X_train_std = np.std(self.X_train_, axis=0)

            self._y_train_mean = np.mean(self.y_train_, axis=0)
            self._y_train_std = np.std(self.y_train_, axis=0)

        else:
            self._X_train_mean = np.atleast_2d(0.)
            self._X_train_std = 1.
            
            self._y_train_mean = np.atleast_2d(0.)
            self._y_train_std = 1.

class MultiFidelityRBF(BaseRBF):
    ''' Carry out Radial Basis Regression for single-fielity data

    --- Inputs ---
    kernels: str (default 'cubic')
        Basis function used to for RBF computation, 
        ['cubic', 'thin_plate', 'gaussian']
    param: float (default None)
        Value of parameter in certain interpolation basis functions. 
        If basis function has no parameter, value is ignored. If 
        optimize == True, value is used as initial guess for optimizer.
        If none, value is defaulted.
    optimize: bool (default True)
        If True, optimize hyperparameter using leave-one-out 
        cross-validation.
    optimizer: str (default 'LBFGS-B')
        Type of optimizer to use for hyperpatameter solution. Must
        correspond to options in scipy.minimize
    order: int (default None)
        Order of supplemental polynomial. If -1, no supplemental
        polynomial is used. If None, value is defaulted based on 
        basis function type
    normalize: bool (default True)
        If True, normalize training data prior to fitting model
    alpha: float (default 1e-10)
        Value of ridge regularization parameter
    n_restarts: None (default 10)
        Number of times to restart gradient-based optimizer during 
        model training. If None, no restarts are performed
    '''
    def __init__(self, kernel='cubic', param=None, optimize=True,
                order=None, normalize=True, alpha=1e-10,
                optimizer='SLSQP', n_restarts=None, n_jobs=1):
        super().__init__(kernel=kernel, param=param, optimize=optimize,
                order=order, normalize=normalize, alpha=alpha,
                optimizer=optimizer, n_restarts=n_restarts, n_jobs=n_jobs)
        
    def fit(self, x_lf, y_lf, x_hf, y_hf):
        ''' Train Multifidelity RBF regressor

        --- Inputs ---
        x_lf: (n_samples_lf, n_features) numpy.ndarray
            Array of low-fidelity inputs
        y_lf: (n_samples_lf, n_responses) numpy.ndarray
            Array of low-fidleity responses
        x_hf: (n_samples_hf, n_features) numpy.ndarray
            Array of high-fidelity inputs
        y_hf: (n_samples_hf, n_responses) numpy.ndarray
            Array of high-fidelity responeses
        ''' 
        def leave_one_out(x):
            if self.optimize and self.kernel in ['gaussian']:
                rho = x[0]
                param = x[1:]
            else:
                rho = x
                param = None

            A = self._evaluate_basis(X_train, X_train, self.kernel, param)

            if self.order >= 0:
                F = self._evaluate_poly(X_train, self.order)
                A = np.vstack((np.hstack((A,F)),
                        np.hstack((F.T, np.zeros((F.shape[1], F.shape[1]))))))

            A += np.eye(A.shape[0])*self.alpha
            b = y_train - rho*self.lf_regressor.predict(X_train)

            if self.order >= 0:
                b = np.vstack((b, np.zeros((F.shape[1], y_train.shape[1]))))

            coeffs = la.solve(A,b)
            betas = coeffs[0:X_train.shape[0],:]

            # Invert A using cholesky decomposition
            # L = la.cholesky(A, lower=True)
            # L_inv = la.solve_triangular(L.T, np.eye(L.shape[0]))
            # A_inv = np.matmul(L_inv, L_inv.T)
            A_inv = la.inv(A)

            LOO = 0.0
            for i in range(X_train.shape[0]):
                beta = betas[i,:]
                phi_inv = A_inv[i,i]
                LOO += np.sum((beta/phi_inv)**2.)

            return LOO

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

        # Set default kernel and order if either are not provided
        if self.kernel is None:
            self.kernel = 'thin_plate'

        if self.order is None:
            if self.kernel == 'gaussian':
                self.order = -1

            elif self.kernel in ['cubic', 'thin_plate']:
                self.order = 1

            # FIX: If other kernels are provided, set default orders here

        if self.param is None:
            if self.kernel == 'gaussian':
                # self.param = X_train.mean(axis=0)
                self.param = np.ones((X_train.shape[1],))

            elif self.kernel in ['cubic', 'thin_plate']:
                self.param = None

            else:
                self.param = 1.0

            # FIX: If other kernels are provided, set default params here

        # Train low-fidelity model
        self.lf_regressor = SingleFidelityRBF(
            kernel=self.kernel,
            param=None,
            optimize=self.optimize,
            optimizer=self.optimizer,
            normalize=self.normalize,
            alpha=self.alpha,
            n_restarts=self.n_restarts
        )
        self.lf_regressor.fit(X_train_lf, y_train_lf)
        
        # Optimize hyperparameters
        if self.optimizer is None:
            self.optimizer = 'L-BFGS-B'

        if self.optimizer.lower() == 'ga':
            pass

        else:
            if self.n_restarts is None:
                self.n_restarts = 0

            rho_test = np.logspace(-2, 2, self.n_restarts).reshape((-1,1))
            if self.optimize and self.kernel in ['gaussian']:
                param_test = np.logspace(-4,4,self.n_restarts)
                param_test = np.vstack((param_test,)*X_train.shape[1]).T
                x_test = np.hstack((rho_test, param_test))
                bounds = ((1e-2,1e2),) + ((1e-5,1e5),)*(x_test.shape[1]-1)
                
            else:
                x_test = rho_test
                bounds = ((1e-2,1e2),)

            opt_param = []
            opt_obj = []
            for p in x_test:
                opt = minimize(
                    fun=leave_one_out,
                    x0=p,
                    bounds=bounds,
                    tol=1e-10,
                    jac='3-point',
                    options={'maxiter': 100, 'disp': False}
                )

                if opt.success:
                    opt_param.append(opt.x)
                    opt_obj.append(opt.fun)

            x_star = opt_param[np.argmin(opt_obj)]
            if self.optimize and self.kernel in ['gaussian']:
                self.rho_ = x_star[0]
                self.param = x_star[1:]
            else:
                self.rho_ = x_star.item(0)

        # Construct traning matrices
        A = self._evaluate_basis(X_train, X_train, self.kernel, self.param)

        if self.order >= 0:
            F = self._evaluate_poly(X_train, self.order)
            A = np.vstack((np.hstack((A,F)),
                        np.hstack((F.T, np.zeros((F.shape[1], F.shape[1]))))))
        A += np.eye(A.shape[0])*self.alpha

        b = y_train - self.rho_*self.lf_regressor.predict(X_train)

        if self.order >= 0:
            b = np.vstack((b, np.zeros((F.shape[1], y_train.shape[1]))))

        coeffs = la.solve(A,b)

        # Store alphas and betas
        self.betas_ = coeffs[0:X_train.shape[0],:]

        if self.order >= 0:
            self.alphas_ = coeffs[X_train.shape[0]:,:]

    def predict(self, X_test):
        ''' Make predictions using multi-fidelity regressor

        --- Inputs ---
        X_test: (n_test_pts, n_features) numpy.ndarray
            Array of test points to evaluate

        --- Returns ---
        Y: (n_test_pts, 1) numpy.ndarray
            Array of mean prediction values at test points
        '''
        X_train = self._apply_normalization(
            self.X_train_, self._X_train_mean, self._X_train_std
        )
        X_test = self._apply_normalization(
            X_test, self._X_train_mean, self._X_train_std
        )

        Phi = self._evaluate_basis(
            X_train, X_test, self.kernel, self.param
        )
        y_lf = self.lf_regressor.predict(X_test)
        y = np.matmul(Phi.T,self.betas_)
        y += self.rho_*y_lf

        if self.order >= 0:
            Q = self._evaluate_poly(X_test, self.order)
            
            y += np.matmul(Q, self.alphas_)

        y = self._undo_normalization(y, self._y_train_mean, self._y_train_std)

        return y

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
    

class SingleFidelityLocalRBF(BaseRBF):
    ''' Train an evaluate RBF models fit using only the k-Nearest-Neighbors
        of the evaluation points
    '''
    def __init__(self, kernel='cubic', param=None, optimize=True,
                order=None, normalize=True, alpha=1e-10,
                optimizer='SLSQP', n_restarts=None, k_neighbors=30,
                leafsize=100, n_jobs=1):
        super().__init__(kernel=kernel, param=param, optimize=optimize,
                order=order, normalize=normalize, alpha=alpha,
                optimizer=optimizer, n_restarts=n_restarts, n_jobs=n_jobs)
        self.k_neighbors = k_neighbors
        self.leafsize = leafsize

    def fit(self, X, y):
        ''' Set up single fidelity data for local RBF regression

        --- Inputs ---
        X: (n_samples, n_features) numpy.ndarray
            Array of inputs
        y: (n_samples, n_responses) numpy.ndarray
            Array of low-fidelity responses
        ''' 

        # Store training parameters
        if len(y.shape) < 2:
            self.y_train_ = y.reshape((-1,1))
        else:
            self.y_train_ = y

        if len(X.shape) < 2:
            self.X_train_ = X.reshape((-1,1))
        else:
            self.X_train_ = X

        self.n_samples, self.n_features = self.X_train_.shape

        # Set default kernel, order and param if either are not provided
        if self.kernel is None:
            self.kernel = 'thin_plate'

        if self.order is None:
            if self.kernel == 'gaussian':
                self.order = -1

            elif self.kernel in ['cubic', 'thin_plate']:
                self.order = 1

            # FIX: If other kernels are provided, set default orders here

        if self.param is None:
            if self.kernel == 'gaussian':
                # self.param = X_train.mean(axis=0)
                self.param = np.ones((self.X_train_.shape[1],))

            else:
                self.param = 1.0

        # Train KD Tree for NN search
        self.tree = KDTree(self.X_train_, leaf_size=self.leafsize)

    def predict(self, X_test):
        ''' Evaluate points in X 

        --- Inputs ---
        X_test: (n_test, n_features):
            Array of evaluation points

        --- Return ---
        y_test: (n_test, n_response):
            Array of predictions
        '''
        
        # Define single point evaluation point for parallel loop
        def evaluate_single_point(ind):
            pt = np.atleast_2d(X_test[ind,:])
            nn_inds = self.tree.query(pt, self.k_neighbors, return_distance=False)

            X_subset = self.X_train_[sorted(nn_inds.flatten()),:]
            y_subset = self.y_train_[sorted(nn_inds.flatten()),:]

            local_model = SingleFidelityRBF(
                kernel=self.kernel,
                param=self.param,
                optimize=self.optimize,
                order=self.order,
                normalize=self.normalize,
                alpha=self.alpha,
                optimizer=self.optimizer,
                n_restarts=self.n_restarts
            )
            local_model.fit(X_subset, y_subset)
            predictions[ind,:] = local_model.predict(pt)

        # Set up number of cpus
        if self.n_jobs < 0:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = self.n_jobs

        # Loop over test points in X_test usin gparallel pool
        predictions = np.zeros((X_test.shape[0], self.y_train_.shape[1]))
        with Parallel(n_jobs=num_workers, require='sharedmem') as parallel:
            accumulator = 0
            n_iter = 0
            
            while accumulator < X_test.shape[0]:
                begin = accumulator
                end = min(accumulator+num_workers, X_test.shape[0])
                
                results = parallel(delayed(evaluate_single_point)(i) \
                                                    for i in range(begin, end))

                n_iter += 1
                accumulator += len(results) 

        return predictions
