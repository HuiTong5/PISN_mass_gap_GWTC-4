import os
import sys
sys.path.append('/home/hui.tong/projects/PISN_GWTC_4/analysis-scripts/')
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from bilby.core.utils import logger

import bilby
import gwpopulation
gwpopulation.set_backend('jax')
from gwpopulation.utils import truncnorm
from gwpopulation_pipe.data_analysis import resample_single_event_posteriors
from gwpopulation.utils import to_numpy
import pandas as pd
from population_inference import *

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
    # sys.path.append('models/default_spin/')
    sys.path.append('/home/hui.tong/projects/PISN_GWTC_4/models/default_spin/')
    # sys.path.append('models/transition_spin/')
    sys.path.append('/home/hui.tong/projects/PISN_GWTC_4/models/transition_spin/')
    result = bilby.read_in_result(file)

    module = get_module(model, use_default_models=True, effective_spin=effective_spin)
    likelihood = setup_likelihood(module, seed, maximum_uncertainty, exclude, effective_spin, GWTC_3)
    priors = get_priors(module.priors, conversion=conversion)

    _, events = load_posteriors(exclude)

    result.meta_data["event_ids"] = events

    print("Start to resample_single_event_posteriors")
    resample_single_event_posteriors(likelihood, result, save=True)

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
