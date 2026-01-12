import os
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'

from popsummary.popresult import PopulationResult
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import bilby

sys.path.append('../../analysis-scripts')
sys.path.append('../../models/default_spin/')

from population_inference import get_priors, get_module, setup_likelihood
from powerlaw1peak2_m2max_no231123 import priors

def get_joint_m1_m2(module, priors, posteriors):
    q = jnp.linspace(0.01, 1, 300)
    m1 = jnp.linspace(3, 200, 300)
    q_grid, m1_grid = jnp.meshgrid(q, m1, indexing = 'ij')

    mass_model=module.models['mass'](normalization_shape=((2000, 4000)))
    mass_model(dict(mass_ratio = q_grid, mass_1 = m1_grid), **{k: posteriors[k][0] for k in priors['mass']})

    pdf = lambda parameters: mass_model(
            dict(mass_ratio = q_grid, mass_1 = m1_grid), **{k: parameters[k] for k in priors['mass']})

    p_2d = jax.lax.map(pdf, posteriors)
    p_2d = jnp.nan_to_num(p_2d)
    p_m1_1d = jnp.trapezoid(p_2d, q, axis=1)

    pos_grid_2D = np.vstack([m1_grid.flatten(), q_grid.flatten()])
    p_2d = p_2d.reshape(p_2d.shape[0], -1)

    return p_2d, pos_grid_2D, p_m1_1d, m1

def get_p_m2(module, priors, posteriors):
    m2_1d = jnp.linspace(3, 200, 300)
    q_grid_1d = jnp.linspace(m2_1d/300, 1, 300).T
    m2_grid_1d=jnp.array([i*jnp.ones(300) for i in m2_1d])
    m1_grid_1d=jnp.array(m2_grid_1d/q_grid_1d)

    mass_model=module.models['mass'](normalization_shape=((2000, 4000)))
    mass_model(dict(mass_ratio = q_grid_1d, mass_1 = m1_grid_1d), **{k: posteriors[k][0] for k in priors['mass']})

    pdf_1d = lambda parameters: mass_model(
            dict(mass_ratio = q_grid_1d, mass_1 = m1_grid_1d), **{k: parameters[k] for k in priors['mass']}) /m1_grid_1d

    ps_1d = jax.lax.map(pdf_1d, posteriors)
    ps_1d = jnp.nan_to_num(ps_1d)

    p_m2_1d = -jnp.array([jnp.trapezoid(ps_1d[i], m1_grid_1d, axis=1) for i in range(ps_1d.shape[0])])

    return p_m2_1d, m2_1d


model = '../../models/default_spin/powerlaw1peak2_m2max_no231123'
result = bilby.read_in_result('../../data_release/results/default_spin/powerlaw1peak2_m2max_no231123_result.hdf5')
module = get_module(model,use_default_models=True,effective_spin=False)

posteriors = {k: jnp.array(result.posterior[k]) for k in priors['mass']}
priors['mass'].pop('H0')
priors['mass'].pop('Om0')

popsummary_result = PopulationResult(
        fname="../../data_release/results/default_spin/powerlaw1peak2_m2max_no231123_popsummary_output.h5",
        hyperparameters=list(result.posterior.keys()),
        hyperparameter_latex_labels=result.get_latex_labels_from_parameter_keys(
            result.posterior.keys()
        ),
    )

popsummary_result.set_hyperparameter_samples(result.posterior, overwrite=True)

p_2d, pos_grid_2D, p_m1_1d, m1_1d = get_joint_m1_m2(module, priors, posteriors)

popsummary_result.set_rates_on_grids('prob_mass_1',
                                     grid_params='mass_1',
                                     positions=np.array(m1_1d),
                                     rates=np.array(p_m1_1d), 
                                     overwrite=True)

popsummary_result.set_rates_on_grids('prob_joint_m1_q',
                                     grid_params=['mass_1', 'mass_ratio'],
                                     positions=np.array(pos_grid_2D),
                                     rates=np.array(p_2d), 
                                     overwrite=True)

p_m2_1d, m2_1d = get_p_m2(module, priors, posteriors)

popsummary_result.set_rates_on_grids('prob_mass_2',
                                     grid_params='mass_2',
                                     positions=np.array(m2_1d),
                                     rates=np.array(p_m2_1d), 
                                     overwrite=True)
