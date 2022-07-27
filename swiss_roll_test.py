''' Test ROM prediction using swiss roll test problem '''
import numpy as np
import matplotlib.pyplot as plt
import sys

from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs
from base_classes import *
from fit_pca import *
from fit_kpca import *
from fit_manifold import *


def swiss_roll(x):
    ''' Compute swiss roll using analytical function
        y = <t*cos(t), h, t*sin(t)>

    --- Inputs ---
    x : (n_samples, 2) numpy.ndarray
        Array of x values

    --- Returns ---
    y : (n_samples, 3)
        Array of response values
    '''
    t = x[:,0]
    h = x[:,1]

    y1 = t*np.cos(t)
    y2 = h
    y3 = t*np.sin(t)

    return np.vstack((y1,y2,y3)).T 

if __name__ == '__main__':
    # Define number of samples, test point, and parameter ranges
    n_train = 1000
    n_test = 10
    x_range = (
        (1.5*np.pi, 4.5*np.pi), 
        (0,20)
    )
    
    # Define ROM settings
    pca_settings = {    # See doc strings in fit_pca.py for details
        'RIC': 1,
        'interp_method': 'rbf',
        'interp_kernel': 'thin_plate',
        'interp_parameter': None,
        'timing': True,
        'random_state': 1,
        'scale_inputs': False,
        'scale_outputs': False,
        'scale_embedding': False,
    }

    kpca_settings = {   # See doc strings in fit_kpca.py for details
        'n_components': None,
        'interp_method': 'rbf',
        'interp_kernel': 'thin_plate',
        'interp_parameter': None,
        'backmap_method': 'auto',
        'backmap_parameters': None,
        'alpha': 1e-6,
        'timing': True,
        'random_state': 1,
        'scale_inputs': False,
        'scale_outputs': False,
        'scale_embedding': False,
        'n_jobs': 1,
    }

    manifold_backmap_settings = {
        'eps': 0.1,
        'p': 4, 
        'cond': False,
        'bound_weights': False,
        'bound_weights_params': None,
    }

    manifold_settings = { # See doc strings in fit_manifold.py
        'method': 'isomap',
        'k_neighbors': 10,
        'n_components': 2,
        'interp_method': 'gauss',
        'interp_kernel': 'matern',
        'interp_parameter': None,
        'backmap_method': 'manifold',
        'backmap_parameters': manifold_backmap_settings,
        'timing': True,
        'random_state': 1,
        'scale_inputs': False,
        'scale_outputs': False,
        'scale_embedding': False,
        'n_jobs': 1,
    }

    # Set up training parameters and evaluate function
    np.random.seed(0)   # Set random seed for lhs sampling
    x_train = lhs(2, samples=n_train)
    range_array = np.atleast_2d(x_range)
    x_train *= range_array[:,1]-range_array[:,0]
    x_train += range_array[:,0]
    y_train = swiss_roll(x_train)

    np.random.seed(1)
    x_test = lhs(2, samples=n_test)
    x_test *= range_array[:,1]-range_array[:,0]
    x_test += range_array[:,0]

    # Fit ROMs
    pcaROM = FitPCA(**pca_settings)
    pcaROM.read_training_params(x_train, param_ranges=x_range)
    pcaROM.read_training_snapshots(y_train)
    pcaROM.execute()
    y_pca = pcaROM.predict(x_test)
    latent_pca = pcaROM._interp_function(x_test)

    kpcaROM = FitKPCA(**kpca_settings)
    kpcaROM.read_training_params(x_train, param_ranges=x_range)
    kpcaROM.read_training_snapshots(y_train)
    kpcaROM.execute()
    y_kpca = kpcaROM.predict(x_test)
    latent_kpca = kpcaROM._interp_function(x_test)
    
    isoROM = FitManifold(**manifold_settings)
    isoROM.read_training_params(x_train, param_ranges=x_range)
    isoROM.read_training_snapshots(y_train)
    isoROM.execute()
    y_iso = isoROM.predict(x_test)
    latent_iso = isoROM._interp_function(x_test)
    
    y_true = swiss_roll(x_test)
    latent_pca_true = pcaROM.fit.transform(y_true)
    latent_kpca_true = kpcaROM.fit.transform(y_true)
    latent_iso_true = isoROM.fit.transform(y_true)

    # Plot 3D
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.scatter(y_train[:,0], y_train[:,1], y_train[:,2], c=x_train[:,0], cmap='jet', marker='.', s=10)
    ax3d.scatter(y_pca[:,0], y_pca[:,1], y_pca[:,2], c='b', marker='*', s=40)
    ax3d.scatter(y_kpca[:,0], y_kpca[:,1], y_kpca[:,2], c='r', marker='*', s=40)
    ax3d.scatter(y_iso[:,0], y_iso[:,1], y_iso[:,2], c='g', marker='*', s=40)
    ax3d.scatter(y_true[:,0], y_true[:,1], y_true[:,2], c='k', marker='*', s=40)
    ax3d.set_title('Three-Dimensional Response', fontsize=18, fontweight='bold')
    ax3d.set_xlabel('X', fontsize=16, fontweight='bold')
    ax3d.set_ylabel('Y', fontsize=16, fontweight='bold')
    ax3d.set_zlabel('Z', fontsize=16, fontweight='bold')
    ax3d.legend(['Train', 'PCA', 'KPCA', 'ISOMAP', 'True'])

    # Plot Embeddings
    figPCA = plt.figure()
    axPCA = figPCA.add_subplot(111)
    axPCA.scatter(pcaROM.embedding[:,0], pcaROM.embedding[:,1], c=x_train[:,0], cmap='jet', marker='.', s=10)
    axPCA.scatter(latent_pca[:,0], latent_pca[:,1], c='b', marker='*', s=40)
    axPCA.scatter(latent_pca_true[:,0], latent_pca_true[:,1], c='k', marker='*', s=40)
    axPCA.set_title('PCA Embedding Space', fontsize=18, fontweight='bold')
    axPCA.set_xlabel('Y1', fontsize=16, fontweight='bold')
    axPCA.set_ylabel('Y2', fontsize=16, fontweight='bold')
    axPCA.legend(['Training Projection', 'Regression', 'True Projection'])

    figKPCA = plt.figure()
    axKPCA = figKPCA.add_subplot(111, projection='3d')
    axKPCA.scatter(kpcaROM.embedding[:,0], kpcaROM.embedding[:,1], kpcaROM.embedding[:,2], c=x_train[:,0], cmap='jet', marker='.', s=10)
    axKPCA.scatter(latent_kpca[:,0], latent_kpca[:,1], latent_kpca[:,2], c='r', marker='*', s=40)
    axKPCA.scatter(latent_kpca_true[:,0], latent_kpca_true[:,1], latent_kpca_true[:,2], c='k', marker='*', s=40)
    axKPCA.set_title('KPCA Embedding Space', fontsize=18, fontweight='bold')
    axKPCA.set_xlabel('Y1', fontsize=16, fontweight='bold')
    axKPCA.set_ylabel('Y2', fontsize=16, fontweight='bold')
    axKPCA.set_zlabel('Y3', fontsize=16, fontweight='bold')
    axKPCA.legend(['Training Projection', 'Regression', 'True Projection'])

    figISO = plt.figure()
    axISO = figISO.add_subplot(111)
    axISO.scatter(isoROM.embedding[:,0], isoROM.embedding[:,1], c=x_train[:,0], cmap='jet', marker='.', s=10)
    axISO.scatter(latent_iso[:,0], latent_iso[:,1], c='g', marker='*', s=40)
    axISO.scatter(latent_iso_true[:,0], latent_iso_true[:,1], c='k', marker='*', s=40)
    axISO.set_title('ISOMAP Embedding Space', fontsize=18, fontweight='bold')
    axISO.set_xlabel('Y1', fontsize=16, fontweight='bold')
    axISO.set_ylabel('Y2', fontsize=16, fontweight='bold')
    axISO.legend(['Training Projection', 'Regression', 'True Projection'])

    # Display plots
    plt.show()
