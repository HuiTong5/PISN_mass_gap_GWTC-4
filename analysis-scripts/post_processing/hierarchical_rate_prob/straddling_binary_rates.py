import os
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import h5py
import numpy as np
import pandas as pd
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import bilby
import gwpopulation
gwpopulation.set_backend('jax')

sys.path.append('../../')
sys.path.append('../../../default')
sys.path.append('../../../models/transition_spin')

from population_inference import get_priors, get_module, setup_likelihood
from powerlaw1peak2_m2gap_fixed_identical import priors as m2gap_priors

def read_in_result_posterior(filename):
    from bilby.core.utils import recursively_load_dict_contents_from_group
    with h5py.File(filename, "r") as ff:
        data = recursively_load_dict_contents_from_group(ff, '/')
    return pd.DataFrame(data["posterior"])

m2gap_priors['mass'].pop('H0')
m2gap_priors['mass'].pop('Om0')

m2gap_model = '../../../models/transition_spin/powerlaw1peak2_m2gap_fixed_identical'
pos_m2gap_result = read_in_result_posterior('../../../data_release/results/transition_spin/powerlaw1peak2_m2gap_fixed_identical_result.hdf5')
m2gap_module = get_module(m2gap_model,use_default_models=True, effective_spin=True)
m2gap_all_priors = get_priors(m2gap_priors)
m2gap_posterior = {k: jnp.array(pos_m2gap_result[k]) for k in m2gap_all_priors}

q = jnp.linspace(0.01, 1, 600)
m1 = jnp.linspace(3, 300, 600) # m1
q_grid, m1_grid = jnp.meshgrid(q, m1, indexing = 'ij')
m2_grid = q_grid * m1_grid

mass_model=m2gap_module.models['mass'](normalization_shape=((1000, 2000)))
mass_model(dict(mass_ratio = q_grid, mass_1 = m1_grid), **{k: pos_m2gap_result[k][0] for k in m2gap_priors['mass']})

pdf_2g1g = lambda parameters: mass_model(
        dict(mass_ratio = q_grid, mass_1 = m1_grid), **{k: parameters[k] for k in m2gap_priors['mass']}) * (m2_grid<=parameters['gap_low']) * (m1_grid>parameters['gap_low']+parameters['gap_width'])

ps = jax.lax.map(pdf_2g1g, m2gap_posterior)

norm = jnp.trapezoid(jnp.trapezoid(ps, m1, axis=2), q, axis=1)[:, None, None]# normalization

rates = norm.T[0,0] * pos_m2gap_result['rate']

np.save('powerlaw1peak2_m2gap_fixed_identical_result_straddling_binary_rate.npy', rates)