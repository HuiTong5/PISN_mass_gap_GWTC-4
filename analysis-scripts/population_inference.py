import argparse
from datetime import datetime
from glob import glob
import importlib
import os
import sys

import bilby
from bilby_pipe.parser import StoreBoolean
import gwpopulation
gwpopulation.set_backend('jax')
from gwpopulation.experimental.jax import NonCachingModel, JittedLikelihood
import h5py
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax_tqdm
import numpy as np
import pandas
import tqdm

from utils import chi_effective_prior_from_isotropic_spins

gwtc2_events = (
    'GW150914_095045', 'GW151012_095443', 'GW151226_033853', 'GW170104_101158',
    'GW170608_020116', 'GW170729_185629', 'GW170809_082821', 'GW170814_103043',
    'GW170818_022509', 'GW170823_131358', 'GW190408_181802', 'GW190412_053044',
    'GW190413_052954', 'GW190413_134308', 'GW190421_213856', 'GW190503_185404',
    'GW190512_180714', 'GW190513_205428', 'GW190517_055101', 'GW190519_153544', 
    'GW190521_030229', 'GW190521_074359', 'GW190527_092055', 'GW190602_175927',
    'GW190620_030421', 'GW190630_185205', 'GW190701_203306', 'GW190706_222641',
    'GW190707_093326', 'GW190708_232457', 'GW190719_215514', 'GW190720_000836',
    'GW190725_174728', 'GW190727_060333', 'GW190728_064510', 'GW190731_140936',
    'GW190803_022701', 'GW190805_211137', 'GW190828_063405', 'GW190828_065509', 
    'GW190910_112807', 'GW190915_235702', 'GW190924_021846', 'GW190925_232845',
    'GW190929_012149', 'GW190930_133541'
)

gwtc3_events = (
    'GW191103_012549', 'GW191105_143521',
    'GW191109_010717', 'GW191127_050227', 'GW191129_134029', 'GW191204_171526',
    'GW191215_223052', 'GW191216_213338', 'GW191222_033537', 'GW191230_180458',
    'GW200112_155838', 'GW200128_022011', 'GW200129_065458', 'GW200202_154313', 
    'GW200208_130117', 'GW200209_085452', 'GW200216_220804', 'GW200219_094415',
    'GW200224_222234', 'GW200225_060421', 'GW200302_015811', 'GW200311_115853',
    'GW200316_215756',
)


def effective_spin_prior(data):
    return data['prior'] * 4 * chi_effective_prior_from_isotropic_spins(
        data["chi_eff"], data["mass_ratio"]
    )


def load_posteriors(exclude = [], GWTC_3=False, save = False):
    files = sorted([
        glob(f'data/posteriors/GWTC-2.1/*{event}*_cosmo.h5')[0]
        for event in gwtc2_events
    ])

    files += sorted([
        glob(f'data/posteriors/GWTC-3/*{event}*_cosmo.h5')[0]
        for event in gwtc3_events
    ])

    if not GWTC_3:
        files += sorted(glob(
            '/home/rp.o4/catalogs/GWTC-4/GWTC4-Stable_Release-1/4c4fd2cef_717/bbh_only/*.hdf5',
        ))

    posteriors = []
    events = []
    labels = ['C00:NRSur7dq4', 'C01:Mixed', 'C00:Mixed']
    for file in tqdm.tqdm(files):
        filename = file.split('/')[-1]

        posterior = {}

        with h5py.File(file, 'r') as f:
            label = None
            for _label in labels:
                if _label in f.keys():
                    label = _label
                    break
            if label is None:
                for key in f.keys():
                    if "posterior_samples" in f[key]:
                        label = key
                        break
            data = f[label]['posterior_samples'][:]

            for key in (
                'mass_1_source', 'mass_ratio', 'redshift',
                'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2',
                'chi_eff', 'chi_p',
            ):
                posterior[key] = data[key]
            posterior['mass_1'] = posterior['mass_1_source']
            del posterior['mass_1_source']
        # priors are:
        # - flat in detectors frame component masses
        # - uniform in Kerr parameters
        # - isotropic in spin directions
        # - comoving redshift prior
        # normalization constants don't matter so just ignore them
        # include Jacobian from:
        # - detector-frame masses -> source frame masses
        # - secondary mass to mass ratio
        posterior['prior'] = (
            bilby.gw.prior.UniformSourceFrame(
                minimum = posterior['redshift'].min(),
                maximum = posterior['redshift'].max(),
                name = 'redshift',
            ).prob(posterior['redshift'])
            * (1 + posterior['redshift']) ** 2
            * posterior['mass_1']
        ) / 4

        posterior['prior_effective_spin'] = effective_spin_prior(posterior)
        event = 'GW' + file.split('GW')[-1].split('-')[0].split('_PE')[0]
        if event in exclude:
            print('excluding', file)
            print(event, 'in exclude')

        elif posterior['mass_1'].max() < 3:
            print('excluding', file)
            print(event, 'maximum mass_1_source < 3')

        elif (posterior['mass_1'] * posterior['mass_ratio']).max() < 3:
            print('excluding', file)
            print(event, 'maximum mass_2_source < 3')

        else:
            posteriors.append(pandas.DataFrame.from_dict(posterior))
            events.append(event)

    return posteriors, events


def load_injections(GWTC_3 = False, far_cut = 1, snr_cut = 10, save = False):

    if GWTC_3:
        print("Using GWTC-3 injection")
        file = (
            'data/injections/mixture-semi_o1_o2-real_o3-cartesian_spins_20250503134659UTC.hdf'
        )
    else:
        print("Using GWTC-4 injection")
        file = (
            'data/injections/'
            'mixture-semi_o1_o2-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf'
        )

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


def get_model(models):
    if type(models) is dict:
        models = list(models.values())
    if type(models) not in (list, tuple):
        models = [models]
    return NonCachingModel(
        [model() if type(model) is type else model for model in models]
    )

def chi_eff_width(parameters):

    parameters['chi_eff_width'] = parameters['chi_eff_max_high'] - parameters['chi_eff_min_high']

    return parameters

def get_priors(priors, conversion=False):
    all_priors = {}
    for par in priors:
        for k in priors[par]:
            all_priors[k] = priors[par][k]
    if conversion:
        print('Using conversion for chi_eff width')
        return bilby.prior.ConditionalPriorDict(all_priors, conversion_function=chi_eff_width)
    else:
        return bilby.prior.ConditionalPriorDict(all_priors)


def get_likelihood(models, posteriors, injections, maximum_uncertainty):
    selection_function = gwpopulation.vt.ResamplingVT(
        model = get_model(models),
        data = injections,
        n_events = len(posteriors),
        marginalize_uncertainty = False,
        enforce_convergence = False,
    )

    return gwpopulation.hyperpe.HyperparameterLikelihood(
        posteriors = posteriors,
        hyper_prior = get_model(models),
        selection_function = selection_function,
        maximum_uncertainty = maximum_uncertainty,
    )


def run_sampler(likelihood, priors, label, outdir, **kwargs):
    default_kwargs = dict(
        sampler = 'dynesty',
        plot = False,
        check_point_plot = False,
        resume = True,
        nlive = 1500,
        dlogz = 0.1,
        sample = 'acceptance-walk',
        naccept = 15,
    )

    for kwarg in kwargs:
        default_kwargs[kwarg] = kwargs[kwarg]

    return bilby.run_sampler(
        likelihood = likelihood,
        priors = priors,
        label = label,
        outdir = outdir,
        use_ratio = True,
        save = 'hdf5',
        **default_kwargs,
    )


def get_module(model, use_default_models, effective_spin):
    path_to_repo = '/'.join(__file__.split('/')[:-2])
    path_in_repo_and_label = model.split('.py')[0].split('/')
    path_in_repo = '/'.join(path_in_repo_and_label[:-1])
    label_at_path = path_in_repo_and_label[-1]

    print('path to repo:', path_to_repo)
    print('path in repo:', path_in_repo)
    print('label at path:', label_at_path)

    sys.path.append(path_to_repo + '/' + path_in_repo)
    module = importlib.import_module(label_at_path)

    sys.path.append(path_to_repo + '/default')
    default = importlib.import_module('default_id')

    if use_default_models:
        for par in ['redshift']:
            if (par not in module.models) or (par not in module.priors):
                assert par not in module.models
                assert par not in module.priors
                print(f'{par} model missing - replacing with default/default_id')
                module.models[par] = default.models[par]
                module.priors[par] = default.priors[par]

        if effective_spin:
            par = 'chi_eff'
            if (
                ('chi_eff' not in module.models)
                and ((par not in module.models) or (par not in module.priors))
            ):
                assert par not in module.models
                assert par not in module.priors
                print(f'{par} model missing - replacing with default/default_id')
                module.models[par] = default.models[par]
                module.priors[par] = default.priors[par]
        else:
            for par in 'spin_mags', 'spin_tilts':
                if (
                    ('chi_eff' not in module.models)
                    and ((par not in module.models) or (par not in module.priors))
                ):
                    assert par not in module.models
                    assert par not in module.priors
                    print(f'{par} model missing - replacing with default/default_id')
                    module.models[par] = default.models[par]
                    module.priors[par] = default.priors[par]

    assert module.models.keys() == module.priors.keys()

    return module


def setup_likelihood(
    module, seed, maximum_uncertainty, exclude, effective_spin, GWTC_3
):
    print('loading posteriors...')
    posteriors, events = load_posteriors(exclude, GWTC_3)

    print('loading injections...')
    injections = load_injections(GWTC_3)

    if effective_spin:
        assert 'spin_mags' not in module.models
        assert 'spin_mags' not in module.priors
        assert 'spin_tilts' not in module.models
        assert 'spin_tilts' not in module.priors
        assert 'chi_eff' in module.models
        assert 'chi_eff' in module.priors
        
        for posterior in posteriors:
            posterior['prior'] = posterior.pop('prior_effective_spin')
        injections['prior'] = injections.pop('prior_effective_spin')

    np.random.seed(int(seed))

    likelihood = get_likelihood(
        module.models, posteriors, injections, float(maximum_uncertainty)
    )

    return likelihood


def likelihood_extras(key, parameters, likelihood):
    extras = likelihood.generate_extra_statistics(parameters)
    max_variance = jnp.array(
        [extras[f'var_{i}'] for i in range(likelihood.n_posteriors)]
    ).max()
    min_neff = 1 / (
        max_variance + 1 / likelihood.samples_per_posterior
    )

    selection_neff = 1 / (
        extras['selection_variance'] / likelihood.n_posteriors**2
        + 1 / likelihood.samples_per_posterior
    )
    
    vt = likelihood.selection_function.surveyed_hypervolume(parameters)
    nexp = jax.random.gamma(key, likelihood.n_posteriors)
    rate = nexp / extras['selection'] / vt

    ln_lkl = jnp.array(
        [extras[f'ln_bf_{i}'] for i in range(likelihood.n_posteriors)],
    ).sum()
    ln_lkl -= likelihood.n_posteriors * jnp.log(extras['selection'])

    return dict(
        log_likelihood = ln_lkl,
        rate = rate,
        selection = extras['selection'],
        selection_variance = extras['selection_variance'],
        variance = extras['variance'],
        min_neff = min_neff,
        selection_neff = selection_neff,
    )


def postprocess(seed, result, likelihood):
    n = len(result.posterior)
    keys = jax.random.split(jax.random.key(seed), n)
    posterior = {k: jnp.array(v) for k, v in result.posterior.items()}

    @jax_tqdm.scan_tqdm(n, print_rate = 1, tqdm_type = 'std')
    @jax.jit
    def single(carry, x):
        i, key, parameters = x
        extras = likelihood_extras(key, parameters, likelihood)
        return carry, extras

    extras = jax.lax.scan(single, None, (jnp.arange(n), keys, posterior))[1]
    ln_lkl = extras.pop('log_likelihood', None)

    for k in extras:
        result.posterior[k] = np.array(extras[k])

    result.save_to_file(overwrite = True, extension = 'hdf5')

    return result


def prior_fraction(likelihood, priors, n = 50_000):
    if type(likelihood) is JittedLikelihood:
        likelihood = likelihood._likelihood

    samples = priors.sample(n)
    for k in samples:
        samples[k] = jnp.array(samples[k])

    @jax_tqdm.scan_tqdm(n, print_rate = 1, tqdm_type = 'std')
    @jax.jit
    def single(carry, x):
        i, parameters = x
        likelihood.parameters.update(parameters)
        ln_lkls, variances = likelihood.ln_likelihood_and_variance()
        return carry, variances

    variances = jax.lax.scan(single, None, (jnp.arange(n), samples))[1]

    w = variances < likelihood.maximum_uncertainty
    frac = w.mean()
    error = ((jnp.mean(w**2) - frac**2) / n)**0.5

    return frac, error


def save_result(result, likelihood, priors):
    fraction, fraction_error = prior_fraction(likelihood, priors)
    ln_fraction = np.log(fraction)
    ln_fraction_error = fraction_error / fraction

    with open(result.outdir + '/' + result.label + '.txt', 'w') as f:
        f.write(f'model name: {result.label.split("-")[0]}\n')
        f.write(f'path to result directory on CIT: {result.outdir}\n')
        f.write(f'ln(evidence): {result.log_bayes_factor} +/- {result.log_evidence_err}\n')
        f.write(f'ln(prior fraction): {ln_fraction} +/- {ln_fraction_error}\n')
        f.write(f'overall ln(evidence): {result.log_bayes_factor + ln_fraction}\n')


def inference(
    model,
    use_default_models = True,
    resume = False,
    GWTC_3 = False,
    seed = 0,
    maximum_uncertainty = 1,
    exclude = [],
    effective_spin = False,
    conversion = False,
    device = None,
):
    if type(exclude) is str:
        exclude = exclude.split(' ')

    print('model:', model)
    print('seed:', seed)
    print('maximum_uncertainty:', maximum_uncertainty)
    print('exclude:', *exclude)
    print('device:', device)

    if device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    module = get_module(model, use_default_models, effective_spin)

    likelihood = setup_likelihood(
        module, seed, maximum_uncertainty, exclude, effective_spin, GWTC_3
    )

    priors = get_priors(module.priors, conversion=conversion)

    print('initial log likelihood ratio calculation...')
    likelihood.parameters.update(priors.sample())
    likelihood.log_likelihood_ratio()
    print(likelihood.log_likelihood_ratio())
    likelihood = JittedLikelihood(likelihood)
    likelihood.parameters.update(priors.sample())
    print(likelihood.log_likelihood_ratio())

    print('events', likelihood.n_posteriors)
    print('samples per event', likelihood.samples_per_posterior)
    print('found injections', likelihood.selection_function.data['found'])
    print('total injections', likelihood.selection_function.total_injections)
    print('analysis time', likelihood.selection_function.analysis_time)

    outdir = '/'.join(module.__file__.split('/')[:-1]) + '/final_results'

    new_label = '-'.join(
        map(str, (module.__name__, seed, maximum_uncertainty)),
    )

    resume_files = glob(outdir + '/' + new_label + '*_resume.pickle')
    if resume and (len(resume_files) > 0):
        if len(resume_files) > 1:
            print('multiple resume files found - using latest')
        new_label = sorted(resume_files)[-1].split('_resume.pickle')[0].split('/')[-1]
    else:
        new_label += '_' + '-'.join(str(datetime.now()).split(' '))

    print('outdir:', outdir)
    print('label:', new_label)

    print('run bilby...')
    kwargs = getattr(module, 'kwargs', {})
    kwargs['resume'] = resume
    # kwargs['check_point_plot'] = True
    result = run_sampler(likelihood, priors, new_label, outdir, **kwargs)

    print('run postprocess...')
    result = postprocess(int(seed), result, likelihood)

    print('save result...')
    save_result(result, likelihood, priors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--use_default_models', action = StoreBoolean)
    parser.add_argument('--conversion', action = StoreBoolean)
    parser.add_argument('--resume', action = 'store_true')
    parser.add_argument('--GWTC_3', action = 'store_true')
    parser.add_argument('--seed', default = 0)
    parser.add_argument('--maximum_uncertainty', default = 1)
    parser.add_argument('--exclude', default = [])
    parser.add_argument('--effective_spin', action = 'store_true')
    parser.add_argument('--device', default = None)
    inference(**parser.parse_args().__dict__)


if __name__ == '__main__':
    main()
