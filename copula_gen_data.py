# Function set up to generate a dataset for use in model interpolation tests
#
#
# Author: Alex Braafladt
#
# Versions: 4/18/2022 v1 initial creation
#           4/30/2022 v2 mods for repo setup, starting to modularize functions
#
#

# imports
# probabilistic and statistical modeling
import openturns as ot
import openturns.viewer as viewer
from scipy import stats as st
from scipy import special as sp
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.spatial.distance import jensenshannon
from scipy import stats as st
from scipy.stats import multiscale_graphcorr
from scipy.stats import pearsonr
# data and numerical functions
import numpy as np
import pandas as pd
# graphing and visualization functions
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
# order reduction
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
# design of experiments
from doepy import build
# os operations
import os as os
import datetime as dt
import pickle
from joblib import dump, load

# model building
import sys
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs
from base_classes import *
from fit_pca import *
from fit_kpca import *
from fit_manifold import *
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

# custom functions
import stats_functions as sf

# functions to be used by main function (last)


def get_case_dists(param_vect):
    """Returns names, distributions as lists given an input vector"""
    # Gaussian (11) [a: mean, b: stdev]
    g_params = np.array([[5.0, 3.0],
                [45.0, 10.0],
                [param_vect[0],100.0],
                [-350.0,50.0],
                [0.75,0.3],
                [6000.0, 3000.0],
                [7.7, 3.0],
                [35000.0, param_vect[1]],
                [56.0, 10.0],
                [99.0, 10.0],
                [15.0, 3.9]])

    dists = []
    names = []
    for i in range(0,g_params.shape[0]):
        dist = ot.Normal(g_params[i,0],g_params[i,1])
        dists.append(dist)
        names.append(dist.getName())

    # Gaussian mixture (5) [mean1, stdev1, mean2, stdev2, weight1, weight2]
    m_params = np.array([[-1.5, 1.0, 1.5, 1.0, 0.5, 0.5],
                [param_vect[2], 1.0, 5, 3.0, 0.5, 0.5],
                [-1.5, 1.0, 1.5, 1.0, 0.1, 0.9],
                [55.0, 5.0, 85.0, 5.0, param_vect[3], 1.0 - param_vect[3]],
                [0.75, 0.1, 0.9, 0.25, 0.5, 0.5]])

    for i in range(0,m_params.shape[0]):
        mixing = [ot.Normal(m_params[i,0],m_params[i,1]),
                          ot.Normal(m_params[i,2],m_params[i,3])]
        byweights = m_params[i,4:]
        dist = ot.Mixture(mixing, byweights)
        dists.append(dist)
        names.append(dist.getName())

    # Uniform (5)
    u_params = np.array([[param_vect[4],1.0],
                         [0.5,0.6],
                         [35.0,param_vect[5]],
                         [5.0,10.0],
                         [3.0,3.1]])
    for i in range(0,u_params.shape[0]):
        dist = ot.Uniform(u_params[i,0],u_params[i,1])
        dists.append(dist)
        names.append(dist.getName())

    # Beta (5)
    b_params = np.array([[param_vect[6], 0.5, -1.0, 1.0],
                         [5.0, 1.0, -1.0, 1.0],
                         [1.0, 3.0, -1.0, 1.0],
                         [2.0, param_vect[7], -1.0, 1.0],
                         [2.0, 5.0, -1.0, 1.0]])
    for i in range(0,b_params.shape[0]):
        dist = ot.Beta(b_params[i,0],b_params[i,1],b_params[i,2],b_params[i,3])
        dists.append(dist)
        names.append(dist.getName())

    # Gumbel (4)
    e_params = np.array([[param_vect[8], 2.0],
                         [1.0, 2.0],
                         [1.5, 3.0],
                         [1.5, param_vect[9]]])
    for i in range(0,e_params.shape[0]):
        dist = ot.Gumbel(e_params[i,0],e_params[i,1])
        dists.append(dist)
        names.append(dist.getName())
    return dists


def get_joint_dists(doe_series, copula):
    margs = doe_series
    joint = ot.ComposedDistribution(margs, copula)
    return joint


def get_sample_arrays(joint):
    sample = np.array(joint.getSample(1000))
    return sample


def pca_stack(data):
    split = np.split(data,data.shape[0],axis=0)
    flattened = [x.flatten('F') for x in split]
    return np.stack(flattened,axis=1)


def pca_destack(data, sample_dim, dist_dim):
    split = np.split(data,data.shape[1],axis=1)
    reshaped = [np.reshape(x[:,0], (sample_dim, dist_dim), order='F') for x in split]
    return np.stack(reshaped, axis=0)


def format_data_for_global_scaler(data):
    splitcases = np.split(data, data.shape[0], axis=0)
    splitcases = [elem[0] for elem in splitcases] # change this later to use np reshape
    by_dists = np.concatenate(splitcases, axis=0)
    return by_dists


def reshape_data_after_scaling(data, num_cases):
    splitcases = np.split(data, num_cases, axis=0)
    data_formated = np.stack(splitcases, axis=0)
    return data_formated


# main function to call externally to generate dataset


def generate_dataset_copula_interptest(input_ranges, num_cases=1000, train_frac=0.8, seed=42):
    """Specific function for test in exp1 of interpolative models;
    input ranges: np.array, shape=(10,2) dtype=float;
    num_cases: int for total number of simulations;
    train_frac: float, fraction of data to use for training, remainder used for testing;
    seed: value to provide to np random rng setup;
    """
    # set up doe based on input ranges and num_cases -------------------------------------------------------------------
    # distributions that the input variables correspond to
    input_labels = ['GaussianMean',
                    'GaussianVariance',
                    'MixtureMean',
                    'MixtureWeight',
                    'UniformLower',
                    'UniformUpper',
                    'Beta1',
                    'Beta2',
                    'Gumbel1',
                    'Gumbel2']
    # sample design space to develop DoE
    # using latin hypercube
    # first set up parameter labels and ranges in a dict
    input_dict = dict()
    for i in range(0, input_ranges.shape[0]):
        input_dict[input_labels[i]] = input_ranges[i, :]
    # next set the number of cases for the DoE and then create the design
    num_samples = num_cases
    # num_samples = 10000
    lhc_design = build.space_filling_lhs(input_dict, num_samples)
    # simulation setup -------------------------------------------------------------------------------------------------
    doe_setup_result1 = lhc_design.apply(get_case_dists, axis=1)
    vecsize = len(doe_setup_result1[0])
    rng = np.random.default_rng(seed=seed)
    # sample from uniform distribution for correlation and add to correlation matrix for copula
    corr_samp = rng.uniform(low=0.0001, high=1.0, size=vecsize)
    norm_samp = corr_samp / (sum(corr_samp))
    corr = norm_samp * float(vecsize)
    rand_corr = st.random_correlation.rvs(corr, random_state=rng)
    # combine into correlation matrix
    rmat = ot.CorrelationMatrix(vecsize, rand_corr.flatten())
    # make copula object
    cop2 = ot.NormalCopula(rmat)
    # get a list of the joint ot ComposedDistribution objects from a series object with each entry an ot.ComposedDistri
    doe_setup_list = doe_setup_result1.apply(get_joint_dists, copula=cop2).to_list()
    # run simulation ---------------------------------------------------------------------------------------------------
    # apply to all distributions in list from doe
    data_list = list(map(get_sample_arrays, doe_setup_list))
    data_np = np.array(data_list)
    # want to scale across all samples for each output variable (distribution) so first put together
    # the 2d array (concat samples, distributions)
    split_data_by_case = np.split(data_np, data_np.shape[0], axis=0)
    # remove extraneous dimension
    split_data_by_case = [elem[0] for elem in split_data_by_case]
    # combine
    data_np_dists = np.concatenate(split_data_by_case, axis=0)
    # Set up scaling of data
    scaler = StandardScaler()
    scaler.fit(data_np_dists)
    # apply scaling
    scaled_data_np_dists = scaler.transform(data_np_dists)
    # reshape data to 3d array
    split_by_case = np.split(scaled_data_np_dists, data_np.shape[0], axis=0)
    data_np_scaled = np.stack(split_by_case, axis=0)
    # structure data for ecdf hypothesis prior to order reduction ------------------------------------------------------
    # starting from scaled raw data
    raw_data = data_np_scaled
    samp_dim = raw_data.shape[1]
    dist_dim = raw_data.shape[2]
    case_dim = raw_data.shape[0]
    ecdfy_vals = sf.get_ecdf_y(raw_data[0, :, 0])
    data_ecdfx = np.apply_along_axis(sf.get_ecdf_x, axis=1, arr=raw_data)
    # split training and test data -------------------------------------------------------------------------------------
    X = lhc_design
    y = data_ecdfx
    indices = np.arange(X.shape[0])
    x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices,
                                                                                     train_size=train_frac)
    print('Shape of full ecdfx dataset: ', data_ecdfx.shape)
    print('Shape of training ecdfx dataset inputs: ', x_train.shape, ' outputs: ', y_train.shape)
    print('Shape of test ecdfx dataset inputs: ', x_test.shape, ' outputs: ', y_test.shape)
    # generate second set of sample to use in goodness-of-fit testing
    # doe_setup_list_test2 = [doe_setup_list[i] for i in indices_test]
    doe_setup_list_test2 = doe_setup_list
    data_list2 = list(map(get_sample_arrays, doe_setup_list_test2))
    data_np2 = np.array(data_list2)
    data_np_scaled_intermed2 = scaler.transform(format_data_for_global_scaler(data_np2))
    data_np_scaled2 = reshape_data_after_scaling(data_np_scaled_intermed2, data_np2.shape[0])
    # return results from simulating data ------------------------------------------------------------------------------
    return x_train, x_test, y_train, y_test, indices_train, indices_test, scaler, data_np_scaled, data_np_scaled2


def generate_ormap_and_modify_dataset():
    """Specific function to input simulation data from exp1 and output the OR reduced data"""


def generate_goodness_of_fit_results(x_train, y_train, x_test, y_test, ind_train, ind_test, samples_full,
                                     samples_compare, y_train_predicted, y_test_predicted):
    """Generate the metrics used for comparing the regression error and the statistical consistency of
    the data from a ROM=OR+interp example considered"""
    # calculate regression metrics -------------------------------------------------------------------------------------
    # note that this is purely regression analysis, and does not yet consider statistical consistency of data
    # requires 2d arrays, so restacking to format from order reduction modeling
    # mean squared error
    mse_train = mean_squared_error(pca_stack(y_train), y_train_predicted.T, multioutput='raw_values')
    mse_test = mean_squared_error(pca_stack(y_test), y_test_predicted.T, multioutput='raw_values')
    ave_mse_train = mean_squared_error(pca_stack(y_train), y_train_predicted.T, multioutput='uniform_average')
    ave_mse_test = mean_squared_error(pca_stack(y_test), y_test_predicted.T, multioutput='uniform_average')
    # r-squared
    r2_train = r2_score(pca_stack(y_train), y_train_predicted.T, multioutput='raw_values')
    r2_test = r2_score(pca_stack(y_test), y_test_predicted.T, multioutput='raw_values')
    ave_r2_train = r2_score(pca_stack(y_train), y_train_predicted.T, multioutput='uniform_average')
    ave_r2_test = r2_score(pca_stack(y_test), y_test_predicted.T, multioutput='uniform_average')
    # meta-data --------------------------------------------------------------------------------------------------------
    num_samples = samples_full.shape[1]
    num_cases = samples_full.shape[0]
    num_dists = samples_full.shape[2]
    # reformat prediction data to match true data format ---------------------------------------------------------------
    y_test_predicted_formatted = pca_destack(y_test_predicted.T, num_samples, num_dists)
    y_train_predicted_formatted = pca_destack(y_train_predicted.T, num_samples, num_dists)
    # condition predicted ecdfs to be statistically consistent ---------------------------------------------------------
    y_test_predict_ecdfx = np.apply_along_axis(sf.get_monotonic_ecdf_aprox, axis=1, arr=y_test_predicted_formatted)
    y_train_predict_ecdfx = np.apply_along_axis(sf.get_monotonic_ecdf_aprox, axis=1, arr=y_train_predicted_formatted)
    # organize true data for statistical comparison --------------------------------------------------------------------
    # samples
    samples_train_true = samples_full[ind_train,:,:]
    samples_test_true = samples_full[ind_test,:,:]
    samples_train_true_repeat = samples_compare[ind_train,:,:]
    samples_test_true_repeat = samples_compare[ind_test,:,:]
    # ecdfs
    ecdfx_train_true = y_train
    ecdfx_test_true = y_test
    ecdfx_train_true_repeat = np.apply_along_axis(sf.get_ecdf_x, axis=1, arr=samples_train_true_repeat)
    ecdfx_test_true_repeat = np.apply_along_axis(sf.get_ecdf_x, axis=1, arr=samples_test_true_repeat)
    ecdfy_all = sf.get_ecdf_y(samples_full[0,:,0])[1:]
    # epdfs
    num_bins = 50
    epdfp_train_true = np.apply_along_axis(sf.get_epdf_probs, arr=samples_train_true, bins=num_bins, axis=1)
    epdfbin_train_true = np.apply_along_axis(sf.get_epdf_bins, arr=samples_train_true, bins=num_bins, axis=1)
    epdfp_test_true = np.apply_along_axis(sf.get_epdf_probs, arr=samples_test_true, bins=num_bins, axis=1)
    epdfbin_test_true = np.apply_along_axis(sf.get_epdf_bins, arr=samples_test_true, bins=num_bins, axis=1)
    epdfp_train_true_repeat = np.zeros_like(epdfp_train_true)
    epdfbin_train_true_repeat = np.zeros_like(epdfbin_train_true)
    epdfp_test_true_repeat = np.zeros_like(epdfp_test_true)
    epdfbin_test_true_repeat = np.zeros_like(epdfbin_test_true)
    # use true bins from training set
    for i in range(epdfbin_train_true.shape[0]):
        for j in range(epdfbin_train_true.shape[2]):
            bins = epdfbin_train_true[i,:,j]
            epdf_probs = sf.get_epdf_probs(arr=samples_train_true_repeat[i,:,j], bins=bins)
            epdfp_train_true_repeat[i,:,j] = epdf_probs
            epdfbin_train_true_repeat[i,:,j] = bins
    # use true bins from test set
    for i in range(epdfbin_test_true.shape[0]):
        for j in range(epdfbin_test_true.shape[2]):
            bins = epdfbin_test_true[i,:,j]
            epdf_probs = sf.get_epdf_probs(arr=samples_test_true_repeat[i,:,j], bins=bins)
            epdfp_test_true_repeat[i,:,j] = epdf_probs
            epdfbin_test_true_repeat[i,:,j] = bins
    # organize predicted data for statistical comparison ---------------------------------------------------------------
    # ecdfs conditioned
    ecdfx_train_predicted = y_train_predict_ecdfx
    ecdfx_test_predicted = y_test_predict_ecdfx
    # samples
    samples_train_predicted = np.apply_along_axis(sf.sample_ecdf, arr=ecdfx_train_predicted,
                                                  num_samples=num_samples, ecdfy=ecdfy_all, axis=1)
    samples_test_predicted = np.apply_along_axis(sf.sample_ecdf, arr=ecdfx_test_predicted,
                                                  num_samples=num_samples, ecdfy=ecdfy_all, axis=1)
    # epdfs
    epdfp_train_predicted = np.zeros_like(epdfp_train_true)
    epdfbin_train_predicted = np.zeros_like(epdfbin_train_true)
    epdfp_test_predicted = np.zeros_like(epdfp_test_true)
    epdfbin_test_predicted = np.zeros_like(epdfbin_test_true)
    # use true bins from training set
    for i in range(epdfbin_train_true.shape[0]):
        for j in range(epdfbin_train_true.shape[2]):
            bins = epdfbin_train_true[i,:,j]
            epdf_probs = sf.get_epdf_probs(arr=samples_train_predicted[i,:,j], bins=bins)
            epdfp_train_predicted[i,:,j] = epdf_probs
            epdfbin_train_predicted[i,:,j] = bins
    # use true bins from test set
    for i in range(epdfbin_test_true.shape[0]):
        for j in range(epdfbin_test_true.shape[2]):
            bins = epdfbin_test_true[i,:,j]
            epdf_probs = sf.get_epdf_probs(arr=samples_test_predicted[i,:,j], bins=bins)
            epdfp_test_predicted[i,:,j] = epdf_probs
            epdfbin_test_predicted[i,:,j] = bins
    # run statistical differencing tests (KS and JS) -------------------------------------------------------------------
    # storage variables
    ks_train_true_repeat = np.ones((ecdfx_train_true.shape[0],1,ecdfx_train_true.shape[2]))
    ks_test_true_repeat = np.ones((ecdfx_test_true.shape[0],1,ecdfx_test_true.shape[2]))
    ks_train_predicted = np.ones((ecdfx_train_true.shape[0],1,ecdfx_train_true.shape[2]))
    ks_test_predicted = np.ones((ecdfx_test_true.shape[0],1,ecdfx_test_true.shape[2]))
    js_train_true_repeat = np.ones((ecdfx_train_true.shape[0],1,ecdfx_train_true.shape[2]))
    js_test_true_repeat = np.ones((ecdfx_test_true.shape[0],1,ecdfx_test_true.shape[2]))
    js_train_predicted = np.ones((ecdfx_train_true.shape[0],1,ecdfx_train_true.shape[2]))
    js_test_predicted = np.ones((ecdfx_test_true.shape[0],1,ecdfx_test_true.shape[2]))
    # calculations
    # train
    for i in range(ecdfx_train_true.shape[0]):
        for j in range(num_dists):
            # for benchmark true repeat data
            ks_m, ks_p = st.ks_2samp(samples_train_true[i,:,j], samples_train_true_repeat[i,:,j])
            ks_train_true_repeat[i,0,j] = ks_m
            js_d = sf.compute_js_dist(epdfp_train_true[i,:,j], epdfp_train_true_repeat[i,:,j])
            js_train_true_repeat[i,0,j] = js_d
            # for predicted data
            ks_m, ks_p = st.ks_2samp(samples_train_true[i,:,j], samples_train_predicted[i,:,j])
            ks_train_predicted[i,0,j] = ks_m
            js_d = sf.compute_js_dist(epdfp_train_true[i,:,j], epdfp_train_predicted[i,:,j])
            js_train_predicted[i,0,j] = js_d
    # test
    for i in range(ecdfx_test_true.shape[0]):
        for j in range(num_dists):
            # for benchmark true repeat data
            ks_m, ks_p = st.ks_2samp(samples_test_true[i,:,j], samples_test_true_repeat[i,:,j])
            ks_test_true_repeat[i,0,j] = ks_m
            js_d = sf.compute_js_dist(epdfp_test_true[i,:,j], epdfp_test_true_repeat[i,:,j])
            js_test_true_repeat[i,0,j] = js_d
            # for predicted data
            ks_m, ks_p = st.ks_2samp(samples_test_true[i,:,j], samples_test_predicted[i,:,j])
            ks_test_predicted[i,0,j] = ks_m
            js_d = sf.compute_js_dist(epdfp_test_true[i,:,j], epdfp_test_predicted[i,:,j])
            js_test_predicted[i,0,j] = js_d
    # organize differencing data into dataframe ------------------------------------------------------------------------
    # train dfs
    ks_train_true_repeat_df = pd.DataFrame(ks_train_true_repeat.ravel(order='F').reshape((ks_train_true_repeat.shape[2],
                                                                                    ks_train_true_repeat.shape[0])).T.flatten(),
                                     columns=['metric'])
    ks_train_true_repeat_df['testtype'] = 'KS'
    ks_train_true_repeat_df['crossval'] = 'train'
    ks_train_true_repeat_df['comparison'] = 'true'
    ks_train_predicted_df = pd.DataFrame(ks_train_predicted.ravel(order='F').reshape((ks_train_predicted.shape[2],
                                                                                    ks_train_predicted.shape[0])).T.flatten(),
                                     columns=['metric'])
    ks_train_predicted_df['testtype'] = 'KS'
    ks_train_predicted_df['crossval'] = 'train'
    ks_train_predicted_df['comparison'] = 'model'
    js_train_true_repeat_df = pd.DataFrame(js_train_true_repeat.ravel(order='F').reshape((js_train_true_repeat.shape[2],
                                                                                    js_train_true_repeat.shape[0])).T.flatten(),
                                     columns=['metric'])
    js_train_true_repeat_df['testtype'] = 'JS'
    js_train_true_repeat_df['crossval'] = 'train'
    js_train_true_repeat_df['comparison'] = 'true'
    js_train_predicted_df = pd.DataFrame(js_train_predicted.ravel(order='F').reshape((js_train_predicted.shape[2],
                                                                                    js_train_predicted.shape[0])).T.flatten(),
                                     columns=['metric'])
    js_train_predicted_df['testtype'] = 'JS'
    js_train_predicted_df['crossval'] = 'train'
    js_train_predicted_df['comparison'] = 'model'
    # test dfs
    ks_test_true_repeat_df = pd.DataFrame(ks_test_true_repeat.ravel(order='F').reshape((ks_test_true_repeat.shape[2],
                                                                                    ks_test_true_repeat.shape[0])).T.flatten(),
                                     columns=['metric'])
    ks_test_true_repeat_df['testtype'] = 'KS'
    ks_test_true_repeat_df['crossval'] = 'test'
    ks_test_true_repeat_df['comparison'] = 'true'
    ks_test_predicted_df = pd.DataFrame(ks_test_predicted.ravel(order='F').reshape((ks_test_predicted.shape[2],
                                                                                    ks_test_predicted.shape[0])).T.flatten(),
                                     columns=['metric'])
    ks_test_predicted_df['testtype'] = 'KS'
    ks_test_predicted_df['crossval'] = 'test'
    ks_test_predicted_df['comparison'] = 'model'
    js_test_true_repeat_df = pd.DataFrame(js_test_true_repeat.ravel(order='F').reshape((js_test_true_repeat.shape[2],
                                                                                    js_test_true_repeat.shape[0])).T.flatten(),
                                     columns=['metric'])
    js_test_true_repeat_df['testtype'] = 'JS'
    js_test_true_repeat_df['crossval'] = 'test'
    js_test_true_repeat_df['comparison'] = 'true'
    js_test_predicted_df = pd.DataFrame(js_test_predicted.ravel(order='F').reshape((js_test_predicted.shape[2],
                                                                                    js_test_predicted.shape[0])).T.flatten(),
                                     columns=['metric'])
    js_test_predicted_df['testtype'] = 'JS'
    js_test_predicted_df['crossval'] = 'test'
    js_test_predicted_df['comparison'] = 'model'
    # combine dataframes
    metrics_df = pd.concat([ks_train_true_repeat_df, ks_train_predicted_df, js_train_true_repeat_df, js_train_predicted_df,
                            ks_test_true_repeat_df, ks_test_predicted_df, js_test_true_repeat_df, js_test_predicted_df], axis=0)
    # calculate percent of distributions with worse statistical closeness than any true distributions ------------------
    # max values of true-true comparison
    max_ks_train_true = metrics_df[(metrics_df['testtype'] == 'KS') & (metrics_df['crossval'] == 'train')
                                   & (metrics_df['comparison'] == 'true')]['metric'].max()
    max_js_train_true = metrics_df[(metrics_df['testtype'] == 'JS') & (metrics_df['crossval'] == 'train')
                                   & (metrics_df['comparison'] == 'true')]['metric'].max()
    max_ks_test_true = metrics_df[(metrics_df['testtype'] == 'KS') & (metrics_df['crossval'] == 'test')
                                  & (metrics_df['comparison'] == 'true')]['metric'].max()
    max_js_test_true = metrics_df[(metrics_df['testtype'] == 'JS') & (metrics_df['crossval'] == 'test')
                                  & (metrics_df['comparison'] == 'true')]['metric'].max()
    # percent of values of true-model comparison that are above max true-true values -----------------------------------
    per_ks_train_predicted =  metrics_df[(metrics_df['testtype'] == 'KS') & (metrics_df['crossval'] == 'train') &
                                         (metrics_df['comparison'] == 'model') &
                                         (metrics_df['metric'] <= max_ks_train_true)]['metric'].count() / metrics_df[(metrics_df['testtype'] == 'KS') &
                                                                                                           (metrics_df['crossval'] == 'train') &
                                                                                                           (metrics_df['comparison'] == 'model')]['metric'].count()
    per_ks_test_predicted =  metrics_df[(metrics_df['testtype'] == 'KS') & (metrics_df['crossval'] == 'test') &
                                         (metrics_df['comparison'] == 'model') &
                                         (metrics_df['metric'] <= max_ks_test_true)]['metric'].count() / metrics_df[(metrics_df['testtype'] == 'KS') &
                                                                                                           (metrics_df['crossval'] == 'test') &
                                                                                                           (metrics_df['comparison'] == 'model')]['metric'].count()
    per_js_train_predicted =  metrics_df[(metrics_df['testtype'] == 'JS') & (metrics_df['crossval'] == 'train') &
                                         (metrics_df['comparison'] == 'model') &
                                         (metrics_df['metric'] <= max_js_train_true)]['metric'].count() / metrics_df[(metrics_df['testtype'] == 'JS') &
                                                                                                           (metrics_df['crossval'] == 'train') &
                                                                                                           (metrics_df['comparison'] == 'model')]['metric'].count()
    per_js_test_predicted =  metrics_df[(metrics_df['testtype'] == 'JS') & (metrics_df['crossval'] == 'test') &
                                         (metrics_df['comparison'] == 'model') &
                                         (metrics_df['metric'] <= max_js_test_true)]['metric'].count() / metrics_df[(metrics_df['testtype'] == 'JS') &
                                                                                                           (metrics_df['crossval'] == 'test') &
                                                                                                           (metrics_df['comparison'] == 'model')]['metric'].count()
    metric_percents = (per_ks_train_predicted, per_ks_test_predicted,
                       per_js_train_predicted, per_js_test_predicted)
    return ave_mse_train, ave_mse_test, ave_r2_train, ave_r2_test, metrics_df, metric_percents, y_train_predict_ecdfx, y_test_predict_ecdfx, ecdfy_all


def plot_model_statistical_consistency(df_metrics, title_str, closeness_met_tup, mse_r2_list, figsavedir):
    g = sns.catplot(x='testtype',y='metric',data=df_metrics,hue='comparison',palette='Set3',
                    height=10, aspect=0.7, inner='quartile', col='crossval', kind='violin')
    (g.set_axis_labels("", "")
      .set_xticklabels(["Kolmogorov-Smirnov metric", "Jensen-Shannon distance"], size=17)
      .set_titles("{col_name}", size=20)
      .set(ylim=(0,1))
      ._legend.remove())
    #g.figure.suptitle(title_str, y=1.02, fontsize=20)
    plt.legend(fontsize=18)
    # plt.suptitle('Radial Basis Function Surrogate', fontsize=20)
    # add annotations for percent metrics
    ks_train_anno = "{:.0f}".format(100.0*closeness_met_tup[0])+'% of modeled distributions\nas close as farthest true'
    at_ks = AnchoredText(ks_train_anno, prop=dict(size=14), frameon=True, loc='center left')
    at_ks.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    g.axes[0][0].add_artist(at_ks)
    js_train_anno = "{:.0f}".format(100.0*closeness_met_tup[2])+'% of modeled distributions\nas close as farthest true'
    at_js = AnchoredText(js_train_anno, prop=dict(size=14), frameon=True, loc='center right')
    at_js.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    g.axes[0][0].add_artist(at_js)
    ks_test_anno = "{:.0f}".format(100.0*closeness_met_tup[1])+'% of modeled distributions\nas close as farthest true'
    at_ks = AnchoredText(ks_test_anno, prop=dict(size=14), frameon=True, loc='center left')
    at_ks.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    g.axes[0][1].add_artist(at_ks)
    js_test_anno = "{:.0f}".format(100.0*closeness_met_tup[3])+'% of modeled distributions\nas close as farthest true'
    at_js = AnchoredText(js_test_anno, prop=dict(size=14), frameon=True, loc='center right')
    at_js.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    g.axes[0][1].add_artist(at_js)
    # add annotation for MSE and R2 metrics
    train_anno = "MSE = {:.3f}\nR2 = {:.3f}".format(mse_r2_list[0], mse_r2_list[1])
    at = AnchoredText(train_anno, prop=dict(size=14), frameon=True, loc='upper center')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    g.axes[0][0].add_artist(at)
    test_anno = "MSE = {:.3f}\nR2 = {:.3f}".format(mse_r2_list[2], mse_r2_list[3])
    at = AnchoredText(test_anno, prop=dict(size=14), frameon=True, loc='upper center')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    g.axes[0][1].add_artist(at)
    g.figure.savefig(figsavedir+'\\'+title_str+'.png', format='png')
    return g


def plot_selected_distribution(first_data, second_data, case_num=0, dist_num=0):
    """Function to plot selected distribution representations"""
    # select set of data to be on plot
    # ecdf
    selected_first_ecdfx = first_data.scaled_ecdfs[1][case_num, :, dist_num]
    selected_second_ecdfx = second_data.scaled_ecdfs[1][case_num, :, dist_num]
    ecdfy = first_data.scaled_ecdfs[0]
    # epdf
    selected_first_epdfp = first_data.scaled_epdfs[1][case_num, :, dist_num]
    selected_second_epdfp = second_data.scaled_epdfs[1][case_num, :, dist_num]
    selected_epdfbins = first_data.scaled_epdfs[0][case_num, :, dist_num]
    # samples
    selected_first_samples = first_data.scaled_samples[case_num, :, dist_num]
    selected_second_samples = second_data.scaled_samples[case_num, :, dist_num]
    # plot
    fig3, ax3 = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)
    ax = ax3.ravel()
    # ecdf
    ax[0].plot(selected_first_ecdfx, ecdfy, label='first')
    ax[0].plot(selected_second_ecdfx, ecdfy, label='second')
    ax[0].set_title("ECDF comparison for case " + str(case_num) + " distribution " + str(dist_num))
    ax[0].legend()
    ax[0].set_yticks(())
    # histogram
    ax[1].hist(selected_first_samples, bins=selected_epdfbins, alpha=0.3, histtype='stepfilled', label='first')
    ax[1].hist(selected_second_samples, bins=selected_epdfbins, alpha=0.3, histtype='stepfilled', label='second')
    ax[1].set_yticks(())
    ax[1].set_title('Sample comparison')
    # epdf
    center = (selected_epdfbins[:-1] + selected_epdfbins[1:]) / 2.
    width = (selected_epdfbins[1] - selected_epdfbins[0]) * 0.8
    ax[2].bar(center, selected_first_epdfp, width=width, alpha=0.3)
    ax[2].bar(center, selected_second_epdfp, width=width, alpha=0.3)
    ax[2].set_yticks(())
    ax[2].set_title('Empirical PDF')

plot_selected_distribution(data_test, data_comparison_test, 0, 11)