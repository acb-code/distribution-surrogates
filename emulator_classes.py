# Initial setup of object-oriented approach to organize experiment using a probabilistic generative model
# to emulate stochastic distributions as expected from operational simulation using agent-based modeling
# Extension to parametric design context
#
# Author: Alex Braafladt
#
# Version: v1 Initial creation 5/2/2022

# imports
import openturns as ot
import numpy as np
from scipy import stats as st
from sklearn.preprocessing import StandardScaler
from doepy import build

# custom functions
import stats_functions as sf
import copula_gen_data as cpgen


class Simulation:
    """
    Object defining the joint distribution which represents an instance of a simulation

    Attributes
    ----------
    param_vals : np.array
        input values based on simulation configuration (subset of all_param_vals)
    all_param_vals : np.array
        all values used to setup the specifics of the simulation
    corr_rng_seed :
        value to use as the seed for setting up the correlation matrix
    distribution_types : list of strings
        list of the specific types of distributions to be used
    marginals : list of ot distributions
        list of the ot distribution objects that define the marginal distributions
    correlation_matrix : ot.CorrelationMatrix
        correlation matrix between marginals
    copula : ot.NormalCopula
        Gaussian copula created based on the correlation matrix between marginals
    joint_distribution : ot.ComposedDistribution
        the main element of the simulation, combines the marginals through the copula

    Methods
    -------
    See docstrings below

    """
    def __init__(self, input_vals, corr_rng_seed):
        self.input_vals = input_vals
        self.param_vals = []
        self.all_param_vals = []
        self.corr_rng_seed = corr_rng_seed
        self.distribution_types = []
        self.marginals = []
        self.correlation_matrix = None
        self.copula = None
        self.joint_distribution = None

    def update_parameter_values(self):
        """set the values of the input parameters for the simulation"""
        input_params = self.input_vals
        param_list = [[input_params[4] * input_params[7], 3.0], [input_params[5] * 0.8, 10.0],
                      [input_params[0],100.0], [-350.0,input_params[5]+10.*input_params[6]],
                      [input_params[11]+0.3,0.3], [0.1*input_params[0], 1.2*input_params[1]],
                      [input_params[-2], 3.0], [35000.0, input_params[1]], [input_params[-3]*4., 3.9],
                      [-1.5, 1.0, 1.5, 1.0, 1.0 - input_params[4], input_params[4]],
                      [input_params[2], 1.0, 5, 3.0, 0.5, 0.5],
                      [55.0*input_params[4], 5.0, 85.0, 5.0, input_params[3], 1.0 - input_params[3]],
                      [input_params[4]**2, 0.1, 0.9, 0.25, 0.5, 0.5], [input_params[4],1.0],
                      [input_params[-5]*0.5,0.6], [35.0,input_params[5]], [3.0*input_params[6]-1.5,3.1],
                      [input_params[6], 0.5, -1.0, 1.0], [5.0, 1.0*input_params[6], -1.0, 1.0],
                      [2.0, input_params[7], -1.0, 1.0], [input_params[7], 5.0, -1.0, 1.0],
                      [input_params[8], 2.0], [1.0*input_params[6], 2.0], [1.5, 3.0*input_params[6]],
                      [1.5, input_params[9]], [20, input_params[10]], [30, input_params[11]],
                      [input_params[12], 2.], [1, input_params[13]], [input_params[14]]]
        self.param_vals = param_list

    def update_distribution_types(self):
        """Create or replace the list of types of distributions to be included in marginals;
        potentially an input in the future, for now, static
        """
        dist_types = ['gaussian', 'gaussian', 'gaussian', 'gaussian', 'gaussian', 'gaussian',
                      'gaussian', 'gaussian', 'gaussian', 'gaussmix', 'gaussmix', 'gaussmix',
                      'gaussmix', 'uniform', 'uniform', 'uniform', 'uniform', 'beta',
                      'beta', 'beta', 'beta', 'gumbel', 'gumbel', 'gumbel',
                      'gumbel', 'binomial', 'binomial', 'skellam', 'skellam', 'poisson']
        self.distribution_types = dist_types

    def update_marginals(self):
        """Use the param values to fill out the list of marginal distributions"""
        # first, put specific inputs into distribution parameters list
        self.update_parameter_values()
        # then, list all the types of marginal distributions to create
        self.update_distribution_types()
        # then, for each dist_type : param_values pair, create the corresponding ot distribution
        for (vals, name) in zip(self.param_vals, self.distribution_types):
            if name == 'gaussian':
                self.marginals.append(ot.Normal(vals[0], vals[1]))
            elif name == 'gaussmix':
                mix_margs = [ot.Normal(vals[0], vals[1]), ot.Normal(vals[2], vals[3])]
                mix_weights = [vals[4], vals[5]]
                self.marginals.append(ot.Mixture(mix_margs, mix_weights))
            elif name == 'uniform':
                self.marginals.append(ot.Uniform(vals[0], vals[1]))
            elif name == 'beta':
                self.marginals.append(ot.Beta(vals[0], vals[1], vals[2], vals[3]))
            elif name == 'gumbel':
                self.marginals.append(ot.Gumbel(vals[0], vals[1]))
            elif name == 'binomial':
                self.marginals.append(ot.Binomial(int(vals[0]), vals[1]))
            elif name == 'skellam':
                self.marginals.append(ot.Skellam(vals[0], vals[1]))
            elif name == 'poisson':
                self.marginals.append(ot.Poisson(vals[0]))
            else:
                print('Unexpected distribution name')
                self.marginals.append(ot.Normal(0, 1))

    def update_correlation_matrix(self):
        """Use the input random seed and shape of marginals to create the correlation matrix"""
        # requires marginals to be created first
        rng = np.random.default_rng(seed=self.corr_rng_seed)
        num_distributions = len(self.marginals)
        corr_samp = rng.uniform(low=0.0001, high=1.0, size=num_distributions)
        norm_corr_samp = corr_samp / (sum(corr_samp))
        local_corr_mat = norm_corr_samp * float(num_distributions)
        rand_corr_mat = st.random_correlation.rvs(local_corr_mat, random_state=rng)
        self.correlation_matrix = ot.CorrelationMatrix(num_distributions, rand_corr_mat.flatten())

    def update_copula(self):
        """Use the correlation matrix to update the copula object for the simulation"""
        self.update_correlation_matrix()
        self.copula = ot.NormalCopula(self.correlation_matrix)

    def update_joint_distribution(self):
        """Use the marginals and copula to update the joint distribution for the simulation"""
        # create ot distributions for the marginals
        self.update_marginals()
        # create an ot copula to connect the ot distributions
        self.update_copula()
        # combine into a joint distribution
        self.joint_distribution = ot.ComposedDistribution(self.marginals, self.copula)

    def reset_simulation(self):
        """Remove setup of simulation to avoid appending duplicates"""
        self.marginals = []

    def get_joint_distribution_samples(self, n_samples=1000):
        """Get the specified number of random samples from the joint distribution that
        defines the simulation
        """
        self.reset_simulation()
        self.update_joint_distribution()
        samples = np.array(self.joint_distribution.getSample(n_samples))
        return samples


class Data:
    """
    Object to hold different versions of data generated from multiple simulation calls
    -conditions and converts to fill out required data types for Experiment
    -path1: starts from samples and fills out other types
    -path2: starts from ecdf_x and fills out other types

    Attributes
    ----------
    scaled_ecdfs : tuple( ecdfy : np.array 1d, ecdfx : np.array 3d)

    Methods
    -------

    """
    def __init__(self, samples=None, ecdf_vals=None, custom_bins=None, custom_scaler=None, discrete_flags=None):
        self.samples = samples
        self.custom_bins = custom_bins
        self.custom_scaler = custom_scaler
        self.scaler = None
        self.scaled_samples = None
        self.scaled_epdfs = None
        self.scaled_ecdfs = ecdf_vals
        self.discrete_flags = discrete_flags

    def round_samples(self):
        """Round the samples which correspond to discrete distributions (assuming all discrete as int)
        -requires samples set up"""
        if self.discrete_flags is not None:
            list_of_dist_data = np.split(self.samples, axis=2)
            rounded_data_list = []
            for (flag, data_slice) in zip(self.discrete_flags, list_of_dist_data):
                if flag:
                    data_slice = np.around(data_slice)
                rounded_data_list.append(data_slice)
            self.samples = np.stack(rounded_data_list, axis=2)

    def scale_samples(self):
        """Scale samples based on custom scaler input or else sklearn StandardScaler - requires samples set up"""
        if self.custom_scaler is None:
            # use new sklearn standard scaler
            self.scaler = StandardScaler()
            # format to scale across full parametric dataset (scale to the full range)
            samples_formatted_for_scaler = cpgen.format_data_for_global_scaler(self.samples)
            self.scaler.fit(samples_formatted_for_scaler)
            scaled_samples_in_scaler_format = self.scaler.transform(samples_formatted_for_scaler)
            # reformat back to back shape
            self.scaled_samples = cpgen.reshape_data_after_scaling(scaled_samples_in_scaler_format,
                                                                   self.samples.shape[0])
        else:
            self.scaler = self.custom_scaler
            samples_formatted_for_scaler = cpgen.format_data_for_global_scaler(self.samples)
            scaled_samples_in_scaler_format = self.scaler.transform(samples_formatted_for_scaler)
            self.scaled_samples = cpgen.reshape_data_after_scaling(scaled_samples_in_scaler_format,
                                                                   self.samples.shape[0])

    def get_scaled_ecdfs_from_samples(self):
        """Transform scaled samples into ecdf format - requires scaled samples set up"""
        ecdf_y = sf.get_ecdf_y(self.scaled_samples[0,:,0])
        ecdf_x = np.apply_along_axis(sf.get_ecdf_x, axis=1, arr=self.scaled_samples)
        self.scaled_ecdfs = (ecdf_y, ecdf_x)

    def get_scaled_epdfs_from_samples(self):
        """Transform scaled samples into epdf format - requires scaled samples set up"""
        if self.custom_bins is None:
            # determine bins from scaled_samples and hardcoded total number of bins
            num_bins = 50
            epdf_bins = np.apply_along_axis(sf.get_epdf_bins, arr=self.scaled_samples, bins=num_bins, axis=1)
            epdf_probs = np.apply_along_axis(sf.get_epdf_probs, arr=self.scaled_samples, bins=num_bins, axis=1)
        else:
            # bins are given
            bins_to_use = self.custom_bins
            epdf_bins = np.zeros_like(self.scaled_samples)
            epdf_probs = np.zeros_like(self.scaled_samples)
            for i in range(epdf_bins.shape[0]):
                for j in range(epdf_bins.shape[2]):
                    bins = bins_to_use[i,:,j]
                    epdf_bins = bins
                    epdf_probs = sf.get_epdf_probs(arr=self.scaled_samples[i,:,j], bins=bins)
        self.scaled_epdfs = (epdf_bins, epdf_probs)

    def data_setup_from_samples(self):
        """Perform operations to flesh out data starting from a set of samples"""
        # assuming samples already rounded
        self.scale_samples()
        self.get_scaled_ecdfs_from_samples()
        self.get_scaled_epdfs_from_samples()

    def data_setup_from_ecdfs(self):
        """Perform operations to flesh out data starting from a set of ecdfs"""
        raw_ecdf_x_vals = self.scaled_ecdfs[1]
        conditioned_ecdf_vals = np.apply_along_axis(sf.get_monotonic_ecdf_aprox, axis=1, arr=raw_ecdf_x_vals)
        example_ecdf_y = sf.get_ecdf_y(conditioned_ecdf_vals[0,:,0])
        undiscretized_samples = np.apply_along_axis(sf.sample_ecdf, arr=conditioned_ecdf_vals,
                                                    axis=1, num_samples=conditioned_ecdf_vals.shape[1],
                                                    ecdfy=example_ecdf_y)
        # round the discrete samples
        self.samples = undiscretized_samples
        self.round_samples()
        # update the ecdfx values for the discrete distributions
        # todo - specify only the discrete ecdfs to get updated and leave the continuous ones
        self.get_scaled_ecdfs_from_samples()
        self.get_scaled_epdfs_from_samples()


class Experiment:
    """
    Object to hold and operate the Simulation and Data classes to generate the
    full dataset for a given experiment

    Attributes
    ----------

    Methods
    -------

    """
    def __init__(self, param_ranges=None, num_cases=1000):
        self.param_ranges = param_ranges
        self.num_cases = num_cases
        self.doe_table = None
        self.dataset = None
        self.simulations = []
        self.input_labels = []
        self.num_inputs = None

    def set_up_doe(self):
        """Use the parameter ranges and number of cases to generate a design of experiments table"""
        # labels are hardcoded for now, dictionary used by doepy package as lhc input
        self.num_inputs = self.param_ranges.shape[0]
        self.input_labels = []
        doe_input_dict = dict()
        for i in range(self.num_inputs):
            self.input_labels.append('input' + str(i))
            doe_input_dict[self.input_labels[i]] = self.param_ranges[i]
        self.doe_table = build.space_filling_lhs(doe_input_dict, self.num_cases)

    def set_up_simulations(self):
        """Generate simulation objects with inputs based on each design of experiments case"""
        self.simulations = []
        for i in range(self.num_cases):
            case_inputs = self.doe_table.iloc[i, :].to_numpy()
            case_simulation = Simulation(case_inputs, 42)
            self.simulations.append(case_simulation)

    def generate_data(self):
        """Generate data from all simulation objects and compile into Data object"""
















