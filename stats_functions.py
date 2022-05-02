# A collection of the statistical functions used in thesis copula notebooks
#
#
# Author: Alex Braafladt
#
# Versions: 3/27/2022 v1 initial creation
#           4/30/2022 v2 conversion to repo setup
#
#

# imports
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy import stats as st


# statistical distancing functions
def support_intersection(p, q):
    """Collect all the probability values where neither value is zero"""
    sup_int = (
        list(
            filter(
                lambda x: (x[0] != 0) & (x[1] != 0), zip(p, q)
            )
        )
    )
    return sup_int


# separate tuple into two arrays
def get_probs(list_of_tuples):
    """Separate list of tuples into np arrays"""
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q


# condition data arrays and use scipy JS distance function
def compute_js_dist(p, q):
    """Take as input p and q as epdfs with the same edge values
    # check for and discard inconsistent data based on zeros"""
    temp = support_intersection(p, q)
    pn, qn = get_probs(temp)
    return jensenshannon(pn, qn)


# condition data arrays and use scipy KL divergence function - via entropy
def compute_kl_dist(p, q):
    """Compute KL divergence on two np arrays of matched probability values"""
    temp = support_intersection(p, q)
    pn, qn = get_probs(temp)
    return st.entropy(pn, qn)


# statistical object conversion functions

# operation to convert sample arrays to ecdf.y and ecdf.x arrays
def get_ecdf_y(arr):
    """Samples to ECDF.y - function to get the y values of an ecdf given samples in a 1D array"""
    ecdf = ECDF(arr)
    # matching size of ecdfx by removing first entry
    return ecdf.y[1:]


def get_ecdf_x(arr):
    """Samples to ECDF.x - function to get the x values of an ecdf given samples in a 1D array"""
    ecdf = ECDF(arr)
    # removing leading -inf put into array (perhaps for plotting purposes?)
    return ecdf.x[1:]


def get_monotonic_ecdf_aprox(ecdfx_entries):
    """Make the entries in the ecdf monotonically increasing, for now simplified to np.maximum.accumulate"""
    # potentially could make more robust, start at first entry and work forward with nth order approx
    a = np.maximum.accumulate(ecdfx_entries)
    return a


def epdf(samp, n=22, density=True):
    """Samples to EPDF bins and probabilities"""
    if density is True:
        h, e = np.histogram(samp, bins=n, density=density)
        p = h
    else:
        h, e = np.histogram(samp, bins=n, density=density)
        p = h / sum(h)
    return e, p


def get_epdf_probs(arr, bins):
    """Samples to EPDF probabilities alone"""
    _, probs = epdf(arr, n=bins)
    return probs


def get_epdf_bins(arr,bins):
    """Samples to EPDF bins alone"""
    bin_, _ = epdf(arr,n=bins)
    return bin_


def get_ecdf_sample(ecdfx, ecdfy):
    """ECDF to sample - given ecdf points, use linear interpolation to sample the ecdf once"""
    # start from a random uniform sample
    ui = np.random.uniform(0.0,1.0)
    # get indices of ecdf y values around sample
    ind_hi = np.argmax(ecdfy>=ui)
    ind_lo = np.argmax(ecdfy>=ui)-1
    # get ecdf "response" values around ui
    resp_hi = ecdfy[ind_hi]
    resp_lo = ecdfy[ind_lo]
    # get ecdf "input" values around ui
    inpt_hi = ecdfx[ind_hi]
    inpt_lo = ecdfx[ind_lo]
    # fit linear model for continuous approx of ecdf "x" given ui as ecdf "y"
    slope = (inpt_hi-inpt_lo)/(resp_hi-resp_lo)
    intercept = inpt_hi - slope*resp_hi
    sample = slope * ui + intercept
    return sample


# get a specified number of samples from ecdf (realizations of the RV)
def sample_ecdf(ecdfx, num_samples, ecdfy):
    """ECDF to samples - function to get a certain number of samples from an ECDF"""
    samples = np.zeros(num_samples)
    for i in range(num_samples):
        samples[i] = get_ecdf_sample(ecdfx, ecdfy)
    return samples

# data manipulation functions