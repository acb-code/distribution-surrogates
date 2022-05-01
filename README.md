# distribution-surrogates
Set up of surrogate models and order reduction for stochastic distributions

## Outline
### Problem and solution development
1. *1_emulator_basics*: initial overview of need for distributions when using agent-based simulation, overview of emulation approach using joint   distributions defined using parametric marginals and copulas, and an initial joint distribution example
2. *2_toy_data_OR*: initial exploration of statistical tools and unsupervised order reduction modeling tools using a toy dataset from cancer research available through scikit-learn (detailed usage of the techniques used here will be in the following notebooks)
3. *3_emulator_nonparametric_OR*: initial setup for emulator - a generative probabilistic model - considering simulation, structuring data, and testing statistical consistency
4. *4_emulator_parametric_OR*: setup for parametric emulator - using latin hypercube design of experiments - instantiation of order reduction approach with parametric inputs, identification of issues with histogram-based data representation
5. *5_emulator_parametric_ecdf_OR*: demonstration of successful parametric OR with alternative distribition representation using empirical cumulative distribution function
6. *6_emulator_surrogate*: extension of generative model to all distributions being driven by inputs, to include discrete distributions, and for connection to framework for surrogate modeling
7. *7_emulator_interactive_model*: instantiation of interactive model and simulation in tandem to showcase exploration capability with surrogate model

### Functions and tools

* Custom tools:
  * stats_functions
  * copula_gen_data 
* Tools from [1], [2] for surrogate modeling
  * pyToolbox [1]
  * fit_(...).py, hierarchical_kriging, multifidelity_rbf, swiss_roll_test, base_classes [2]

### References:
[ [1] ](https://royalsocietypublishing.org/doi/abs/10.1098/rspa.2021.0495)
[ [2] ](https://smartech.gatech.edu/bitstream/handle/1853/62941/Decker_Aviation2020_Final.pdf?sequence=1)
