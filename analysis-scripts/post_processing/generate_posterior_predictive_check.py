
import os
import sys
sys.path.append('/home/hui.tong/projects/PISN_GWTC_4/analysis-scripts/')
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from bilby.core.utils import logger
from population_inference import *

import bilby
import gwpopulation
gwpopulation.set_backend('jax')
from gwpopulation.utils import truncnorm
from gwpopulation_pipe.common_format import resample_events_per_population_sample, resample_injections_per_population_sample, save_to_common_format
from gwpopulation.utils import to_numpy
import pandas as pd
import dill


def common_format(
    model,
    result_file,
    sample_file,
    use_default_models = True,
    conversion = False,
    GWTC_3 = False,
    seed = 0,
    maximum_uncertainty = 1,
    exclude = [],
    effective_spin = False,
    device = None,
):
    # sys.path.append('../models/default_spin/')
    sys.path.append('/home/hui.tong/projects/PISN_GWTC_4/models/default_spin/')

    module = get_module(model, use_default_models=True, effective_spin=effective_spin)
    likelihood = setup_likelihood(module, seed, maximum_uncertainty, exclude, effective_spin, GWTC_3)

    result = bilby.read_in_result(result_file)
    priors = get_priors(module.priors, conversion=conversion)
    n_samples = len(result.posterior)
    all_samples = dict()
    posterior = result.posterior
    all_samples["posterior"] = posterior

    with open(sample_file, "rb") as ff:
        samples = dill.load(ff)
    if "prior" not in samples["original"]:
        raise
    for key in samples["original"]:
        samples["original"][key] = jnp.asarray(samples["original"][key])

    n_draws = len(samples["names"])
    logger.info(f"Number of draws equals number of events, {n_draws}.")

    logger.info("Generating observed populations.")
    model = get_model(module.models)

    model.parameters.update(priors.sample())
    model.prob(samples["original"])

    observed_dataset = resample_events_per_population_sample(
        posterior=posterior,
        samples=samples["original"],
        model=model,
        n_draws=len(samples["names"]),
    )

    for ii, name in enumerate(samples["names"]):
        new_posterior = pd.DataFrame()
        for key in observed_dataset:
            new_posterior[f"{name}_{key}"] = to_numpy(observed_dataset[key][:, ii])
        all_samples[name] = new_posterior

    logger.info("Generating predicted populations.")
    print('loading injections...')
    model = get_model(module.models)

    model.parameters.update(priors.sample())
    
    injections = load_injections(GWTC_3)
    vt_data = injections
    vt_data.pop("analysis_time")
    vt_data.pop("found")
    vt_data.pop("total_generated")

    model.prob(vt_data)

    synthetic_dataset = resample_injections_per_population_sample(
        posterior=posterior,
        data=vt_data,
        model=model,
        n_draws=n_draws,
    )

    for ii in range(n_draws):
        new_posterior = pd.DataFrame()
        for key in synthetic_dataset:
            new_posterior[f"synthetic_{key}_{ii}"] = to_numpy(
                synthetic_dataset[key][:, ii]
            )
        all_samples[f"synthetic_{ii}"] = new_posterior

    filename = os.path.join(result.outdir, f"{result.label}_full_posterior.hdf5")
    save_to_common_format(
        posterior=all_samples, events=samples["names"], filename=filename
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--result_file')
    parser.add_argument('--sample_file')
    parser.add_argument('--use_default_models', action = StoreBoolean)
    parser.add_argument('--conversion', action = StoreBoolean)
    parser.add_argument('--GWTC_3', action = 'store_true')
    parser.add_argument('--seed', default = 0)
    parser.add_argument('--maximum_uncertainty', default = 1)
    parser.add_argument('--exclude', default = [])
    parser.add_argument('--effective_spin', action = 'store_true')
    parser.add_argument('--device', default = None)
    common_format(**parser.parse_args().__dict__)


if __name__ == '__main__':
    main()
