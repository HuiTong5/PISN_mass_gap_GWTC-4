# Installation

Make sure you have the python requirements in `requirement.txt`. The installation time should no more than a few minutes. This includes JAX which allows the analysis running time ~ minutes to hours depending on sampler settings and GPU hardwares. Analysis using normal numpy with CPU is theoretically possible but strongly not recommended since the analysis might take ~ weeks given the amount of data in GWTC-4.

For analysis using CPU without JAX, please replace all `jax.numpy` by `numpy` and set up the backend of `gwpopulation` by `gwpopulation.set_backend('numpy')`.

# Data collection

Please make sure the appropriate event posterior samples and sensitivity estimate files are downloaded following `collect_samples.sh`.

# Analysis

Run the following command, either on a head node or put it in your job submission script:

```python
python population_inference.py MODEL --use-default-models True (--resume) (--GWTC_3) (--seed 0) (--maximum_uncertainty 1) (--exclude []) (--effective_spin) (--device None)
```

You can run this from anywhere, e.g., if you are in the root of the repo, `python analysis-scripts/population_inference.py ...`.

It has a few required positional argument. The first is the path of the model file you want to run relative to the root of the repo; for the default model this would be `default/default` (the .py is not needed but won't affect anything if it is incldued), i.e., `python population_inference.py example/model`. Also, you should specify whether you want to inherit the models from `default/default.py` by `--use-default-models`.

The optional arguments are in brackets above:
- `resume`: Flag if restarting a run from an automatic checkpoint. You should only need to use this if your run crashed and you don't want to start from scratch.
- `GWTC_3`: Flag if you want to do 69 BBHs GWTC-3 analysis.
-  `conversion`: True if you want to do the spin transition analysis with the flexible high mass spin model.
- `seed`: `np.random.seed(seed)` used at the start.
- `maximum_uncertainty`: Cut applied to the total estimated Monte Carlo variance of the log likelihood.
- `exclude`: Space-separated string of O4a event IDs to exclude from the analysis. The default is to only exclude events if there maximum primary or secondary mass samples are $<3M_\odot$.
- `effective_spin`: Flag to use if running a model in effective spins rather than component spins. At the moment, effective spins analyses are not fully implemented.
- `device`: Used to pin devices to use when running directly on a GPU head node, e.g., `--device 1`, or `--device "0,1,2"`. You could also set this by manually defining the environment variable `CUDA_VISIBLE_DEVICES` before running `python inference.py ...`.

# Results

This will run on the BBHs with FAR<1/year with the 1/rate prior marginalized likelihood. A bilby result file will be produced and save in `MODEL/results/`. Additional [useful quantities from the likelihood](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/hyperpe.py#L236) and [resampled rate posterior draws](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/hyperpe.py#L288) are added to this file in postprocessing.

The fraction of the prior volume that passes the likelihood threshold can be computed using the function `prior_fraction` from [`analysis-scripts/population_inference.py`](../analysis-scripts/population_inference.py'). This will produce a txt file which provide an estimate of the correct evidence in the `overall ln(evidence)` row in the txt file.

# Other examples

## Post-processing

### Generate popsummary files

[Popsummary](https://git.ligo.org/christian.adamcewicz/popsummary/-/tree/main?ref_type=heads) is a Python API for interfacing with standardized LIGO-Virgo-KAGRA Rates and Populations results. 
We use popsummary to store probability density for mass and spin parameters. There are a few examples how to generate popsummary files under [`analysis-scripts/post_processing/popsummary_converter_examples`](../analysis-scripts/post_processing/popsummary_converter_examples). 

The notebook in final results directory would provide instructions how to read popsummary files.

### PE with population-informed priors

Using the inferred population distribution as a prior to inform the masses of individual events, one can use [`/analysis-scripts/post_processing/resample_single_event_posteriors.py`](../analysis-scripts/post_processing/resample_single_event_posteriors.py).

Run the following command:

```python
python resample_single_event_posteriors.py MODEL --file example_result.hdf5 --use-default-models True (--GWTC_3) (--seed 0) (--maximum_uncertainty 1) (--exclude []) (--effective_spin) (--device None)
```

The only additionally required positional argument `--file` is the path of the result file by the analysis script.


### Posterior predictive check

Posterior predictive check is widely used in hierarchical Bayesian inference to check whether the inferred population distribution is consistent with the observation. The script to create samples for posterior predictive check is [`/analysis-scripts/post_processing/generate_posterior_predictive_check.py`](../analysis-scripts/post_processing/generate_posterior_predictive_check.py).

Run the following command:

```python
python resample_single_event_posteriors.py MODEL --file example_result.hdf5 --sample_file example_samples.pkl --use-default-models True (--GWTC_3) (--seed 0) (--maximum_uncertainty 1) (--exclude []) (--effective_spin) (--device None)
```

The additionally required positional arguments: `--file` is the path of the result file by the analysis script and `--sample_file` is the path of the population-informed PE file.

### Hierarchcial merger rates and Bayes factor of hierarchical origin for events

Check out the examples under `/post_processing/hierarchical_rate_prob/`.

`2g1g_hierarchical_merger_rates.py` provides an example to calculate the rates for BBHs with one component in the mass gap.

`2g2g_hierarchical_merger_rates.py` provides an example to calculate the rates for BBHs with both components in the mass gap. Note this is only applicable to m2 notch model where the gap can have finite depth.

`2g1g_hierarchical_merger_prob.py` provides an example to calculate the Bayes factor between the hypotheses that a event has a component in gap and both components are not in the gap.

`calculate_dectetion_fraction.py` provides an example to calculate the fractions of mergers with one component in gap, or both above the gap, or straddling binaries among the detected events assuming a O4a-like sensitivity.