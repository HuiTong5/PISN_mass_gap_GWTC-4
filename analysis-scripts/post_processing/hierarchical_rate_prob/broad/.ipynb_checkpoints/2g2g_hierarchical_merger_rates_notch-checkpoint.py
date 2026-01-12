import os
import sys

# Third-party imports
import numpy as np
import pandas as pd
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import gwpopulation
gwpopulation.set_backend('jax')
from bilby.core.result import read_in_result

# Local path configuration
sys.path.append('/home/hui.tong/projects/PISN_GWTC_4/analysis-scripts')
sys.path.append('/home/hui.tong/projects/PISN_GWTC_4/models/transition_spin')

# Local imports
from population_inference import get_priors, get_module, setup_likelihood
from powerlaw1peak2_m2notch import priors as m2gap_priors

m2gap_priors['mass'].pop('H0')
m2gap_priors['mass'].pop('Om0')

m2gap_model = '/home/hui.tong/projects/PISN_GWTC_4/models/default_spin/powerlaw1peak2_m2notch'
m2gap_result = read_in_result('/home/hui.tong/projects/PISN_GWTC_4/models/default_spin/final_results/powerlaw1peak2_m2notch_broad-0-1_2025-11-08-10:44:06.319261_result.hdf5')
m2gap_module = get_module(m2gap_model,use_default_models=True, effective_spin=False)
m2gap_all_priors = get_priors(m2gap_priors, conversion=False)
m2gap_posterior = {k: jnp.array(m2gap_result.posterior[k]) for k in m2gap_all_priors}

q = jnp.linspace(0.01, 1, 600)
m1 = jnp.linspace(3, 300, 600) # m1
q_grid, m1_grid = jnp.meshgrid(q, m1, indexing = 'ij')
m2_grid = q_grid * m1_grid

mass_model=m2gap_module.models['mass'](normalization_shape=((1000, 2000)))
mass_model(dict(mass_ratio = q_grid, mass_1 = m1_grid), **{k: m2gap_result.posterior[k][0] for k in m2gap_priors['mass']})

pdf_2g2g = lambda parameters: mass_model(
        dict(mass_ratio = q_grid, mass_1 = m1_grid), **{k: parameters[k] for k in m2gap_priors['mass']}) * (m2_grid>parameters['gap_low']) * (m2_grid<=parameters['gap_low']+parameters['gap_width']) * (m1_grid>parameters['gap_low']) * (m1_grid<=parameters['gap_low']+parameters['gap_width'])

ps = jax.lax.map(pdf_2g2g, m2gap_posterior)

norm = jnp.trapezoid(jnp.trapezoid(ps, m1, axis=2), q, axis=1)[:, None, None]# normalization

rates = norm.T[0,0] * m2gap_result.posterior['rate']

np.save('powerlaw1peak2_m2notch_result_2g2g_rate.npy', rates)  # Save norm to a .npy file



