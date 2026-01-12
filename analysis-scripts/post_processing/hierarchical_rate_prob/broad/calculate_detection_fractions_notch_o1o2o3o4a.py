import os

import h5py
import numpy as np
import pandas as pd
import pickle
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import tqdm

import bilby
import gwpopulation
from gwpopulation.utils import to_numpy
gwpopulation.set_backend('jax')

import sys
sys.path.append('/home/hui.tong/projects/PISN_GWTC_4/analysis-scripts')
sys.path.append('/home/hui.tong/projects/PISN_GWTC_4/models/default_spin')
sys.path.append('/home/hui.tong/projects/PISN_GWTC_4/models/transition_spin')

from gwpopulation_pipe.common_format import resample_injections_per_population_sample
from population_inference import get_priors, get_module, setup_likelihood, effective_spin_prior, get_model

def load_injections_o4a(far_cut = 1, snr_cut = 10, save = False):
    file = '/home/rp.o4/offline-injections/mixtures/multirun-mixtures_20250503134659UTC/mixture-semi_o1_o2-real_o3_o4a/mixture-semi_o1_o2-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf'

    injections = {}

    with h5py.File(file, 'r') as f:
        events = f['events'][:]
        fars = [events[key] for key in events.dtype.names if 'far' in key]
        min_fars = np.min(fars, axis = 0)
        snrs = events['semianalytic_observed_phase_maximized_snr_net']
        found = (min_fars < far_cut) | (snrs > snr_cut)
        events = events[found]

        injections['mass_1_source'] = events['mass1_source']
        injections['mass_ratio'] = \
            events['mass2_source'] / injections['mass_1_source']
        injections['redshift'] = events['redshift']
        injections['a_1'] = (
            events['spin1x']**2 + events['spin1y']**2 + events['spin1z']**2
        )**0.5
        injections['a_2'] = (
            events['spin2x']**2 + events['spin2y']**2 + events['spin2z']**2
        )**0.5
        injections['cos_tilt_1'] = events['spin1z'] / injections['a_1']
        injections['cos_tilt_2'] = events['spin2z'] / injections['a_2']

        ln_prior = events[
            'lnpdraw_mass1_source_mass2_source_redshift'
            '_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z'
        ]
        prior = (
            np.exp(ln_prior)
            * injections['mass_1_source']
            * 4 * np.pi**2 * injections['a_1']**2 * injections['a_2']**2
        )
        injections['prior'] = prior / events['weights']

        q = injections['mass_ratio']
        a1 = injections['a_1']
        a2 = injections['a_2']
        c1 = injections['cos_tilt_1']
        c2 = injections['cos_tilt_2']
        s1 = np.sin(np.arccos(c1))
        s2 = np.sin(np.arccos(c2))
        injections['chi_eff'] = (a1 * c1 + q * a2 * c2) / (1 + q)
        injections['chi_p'] = np.max(
            [a1 * s1, q * a2 * s2 * (4 * q + 3) / (4 + 3 * q)], axis = 0,
        )
        injections['prior_effective_spin'] = effective_spin_prior(injections)

        injections['found'] = found.sum()
        injections['total_generated'] = f.attrs['total_generated']

        injections['mass_1'] = injections['mass_1_source']
        del injections['mass_1_source']
        
        for key in 'analysis_time', 'total_analysis_time', 'analysis_time_s':
            if key in f.attrs:
                injections['analysis_time'] = f.attrs[key]
        if 'analysis_time' not in injections:
            print('analysis_time not found')
        else:
            injections['analysis_time'] /= 60 * 60 * 24 * 365.25

    for key in injections:
        injections[key] = jnp.array(injections[key])

    return injections


def read_in_result_posterior(filename):
    from bilby.core.utils import recursively_load_dict_contents_from_group
    with h5py.File(filename, "r") as ff:
        data = recursively_load_dict_contents_from_group(ff, '/')
    return pd.DataFrame(data["posterior"])

model = '/home/hui.tong/projects/PISN_GWTC_4/models/default_spin/powerlaw1peak2_m2notch'
module = get_module(model, use_default_models=True, effective_spin=False)
model = get_model(module.models)

pos_m2gap_result = read_in_result_posterior('/home/hui.tong/projects/PISN_GWTC_4/models/default_spin/final_results/powerlaw1peak2_m2notch_broad-0-1_2025-11-08-10:44:06.319261_result.hdf5')

print('loading injections...')

injections = load_injections_o4a()
vt_data = injections
vt_data.pop("analysis_time")
vt_data.pop("found")
vt_data.pop("total_generated")


model.parameters.update(pos_m2gap_result.iloc[0])
model.prob(vt_data)

n_draws = 300

synthetic_dataset = resample_injections_per_population_sample(
    posterior=pos_m2gap_result,
    data=vt_data,
    model=model,
    n_draws=n_draws,
)

all_categories = dict()

# Vectorized computation over draws and samples
m1 = to_numpy(synthetic_dataset['mass_1'])
q = to_numpy(synthetic_dataset['mass_ratio'])
m2 = m1 * q

gap_low = to_numpy(pos_m2gap_result['gap_low'].values)
gap_width = to_numpy(pos_m2gap_result['gap_width'].values)
gap_high = gap_low + gap_width

# Broadcast gap thresholds to match shape (n_draws, n_samples)
gap_low_2d = gap_low[:, None]
gap_high_2d = gap_high[:, None]

# Masks replicating original logic
mask_one_below_one_in = (m1 > gap_low_2d) & (m1 <= gap_high_2d) & (m2 <= gap_low_2d)
mask_both_in = (m1 > gap_low_2d) & (m1 <= gap_high_2d) & (m2 > gap_low_2d) & (m2 <= gap_high_2d)

# Fractions across samples per draw
all_categories['one_below_gap_one_in_gap'] = mask_one_below_one_in.mean(axis=1)
all_categories['both_in_gap'] = mask_both_in.mean(axis=1)

# Save results to file
output_file = 'fraction_of_different_kinds_detections_notch_o1o2o3o4a_300.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(all_categories, f)

