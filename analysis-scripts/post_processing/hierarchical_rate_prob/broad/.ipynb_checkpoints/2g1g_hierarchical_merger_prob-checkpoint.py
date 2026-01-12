import os
import sys
import argparse

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import pandas as pd
import bilby
import gwpopulation
gwpopulation.set_backend('jax')
from gwpopulation.utils import xp

sys.path.append('/home/hui.tong/projects/PISN_GWTC_4/analysis-scripts')
sys.path.append('/home/hui.tong/projects/PISN_GWTC_4/models/transition_spin')

from population_inference import *
from tqdm.auto import tqdm

def hierarchical_prob(likelihood, samples):
    """
    Resample the original single event posteriors to use the PPD from each
    of the other events as the prior.

    Parameters
    ----------
    samples: pd.DataFrame, dict, list
        The samples to do the weighting over, typically the posterior from
        some run.
    return_weights: bool, optional
        Whether to return the per-sample weights, default = :code:`False`

    Returns
    -------
    new_samples: dict
        Dictionary containing the weighted posterior samples for each of
        the events.
    weights: array-like
        Weights to apply to the samples, only if :code:`return_weights == True`.
    """

    if isinstance(samples, pd.DataFrame):
        samples = [dict(samples.iloc[ii]) for ii in range(len(samples))]
    elif isinstance(samples, dict):
        samples = [samples]
    # weights = xp.zeros((likelihood.n_posteriors, likelihood.samples_per_posterior))
    event_weights = xp.zeros(likelihood.n_posteriors)
    hierarchical_p = []
    for sample in tqdm(samples):
        likelihood.parameters.update(sample.copy())
        likelihood.parameters, added_keys = likelihood.conversion_function(likelihood.parameters)
        new_weights = likelihood.hyper_prior.prob(likelihood.data) / likelihood.sampling_prior
        event_weights += xp.mean(new_weights, axis=-1)
        new_weights = (new_weights.T / xp.sum(new_weights, axis=-1)).T
        hierarchical_weights = new_weights*(likelihood.data['mass_1']>likelihood.parameters['gap_low'])*(likelihood.data['mass_1']<=likelihood.parameters['gap_low']+likelihood.parameters['gap_width'])*(likelihood.data['mass_1']*likelihood.data['mass_ratio']<=likelihood.parameters['gap_low'])
        non_hierarchical_weights = new_weights*(~((likelihood.data['mass_1']>likelihood.parameters['gap_low'])*(likelihood.data['mass_1']<=likelihood.parameters['gap_low']+likelihood.parameters['gap_width'])*(likelihood.data['mass_1']*likelihood.data['mass_ratio']<=likelihood.parameters['gap_low'])))
        hierarchical_p.append(xp.sum(hierarchical_weights, axis=-1)/xp.sum(non_hierarchical_weights, axis=-1))
        # weights += new_weights
        if added_keys is not None:
            for key in added_keys:
                likelihood.parameters.pop(key)
    return hierarchical_p

def post_process(
    model,
    file,
    use_default_models = True,
    conversion = False,
    GWTC_3 = False,
    seed = 0,
    maximum_uncertainty = 1,
    exclude = [],
    effective_spin = False,
    device = None,
):
    if effective_spin:
        print("effective spin")

    injections = load_injections()
    # for file in files:
    # sys.path.append('../models/transition_spin/')
    result = bilby.read_in_result(file)


    module = get_module(model, use_default_models=True, effective_spin=effective_spin)
    likelihood = setup_likelihood(module, seed, maximum_uncertainty, exclude, effective_spin, GWTC_3)
    priors = get_priors(module.priors, conversion=conversion)

    _, events = load_posteriors(exclude)

    result.meta_data["event_ids"] = events

    print("Start to calculate hierarchical prob")
    hierarchical_p = hierarchical_prob(likelihood, result.posterior)
    np.save('powerlaw1peak2_m2gap_fixed_identical_result_hierarchical_p.npy', np.array(hierarchical_p))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--file')
    parser.add_argument('--use_default_models', action = StoreBoolean)
    parser.add_argument('--conversion', action = StoreBoolean)
    parser.add_argument('--GWTC_3', action = 'store_true')
    parser.add_argument('--seed', default = 0)
    parser.add_argument('--maximum_uncertainty', default = 1)
    parser.add_argument('--exclude', default = [])
    parser.add_argument('--effective_spin', action = 'store_true')
    parser.add_argument('--device', default = None)
    post_process(**parser.parse_args().__dict__)


if __name__ == '__main__':
    main()
