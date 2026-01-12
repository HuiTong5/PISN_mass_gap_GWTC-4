import jax
import jax.numpy as jnp
from gwpopulation.utils import powerlaw, truncnorm
from gwpopulation.models.redshift import PowerLawRedshift
import bilby

def spin_mags_model(dataset, mu_a, sigma_a):
    p1 = truncnorm(dataset['a_1'], mu_a, sigma_a, 1, 0)
    p2 = truncnorm(dataset['a_2'], mu_a, sigma_a, 1, 0)
    return p1 * p2


def spin_tilts_model(dataset, mu_cos, sigma_cos, fcos):
    ali1 = truncnorm(dataset['cos_tilt_1'], mu_cos, sigma_cos, 1, -1)
    ali2 = truncnorm(dataset['cos_tilt_2'], mu_cos, sigma_cos, 1, -1)
    iso1 = (-1 <= dataset['cos_tilt_1']) * (dataset['cos_tilt_1'] <= 1) / 2
    iso2 = (-1 <= dataset['cos_tilt_2']) * (dataset['cos_tilt_2'] <= 1) / 2
    return fcos * ali1 * ali2 + (1 - fcos) * iso1 * iso2

def gaussian_chi_eff(dataset, mu_chi_eff, log_sigma_chi_eff):
    return truncnorm(
        dataset["chi_eff"], mu=mu_chi_eff, sigma=10**(log_sigma_chi_eff), low=-1, high=1
    )

models = dict(
    spin_mags = spin_mags_model,
    spin_tilts = spin_tilts_model,
    redshift = PowerLawRedshift(z_max=3.0, cosmo_model="FlatLambdaCDM"),
    chi_eff = gaussian_chi_eff,
)


priors = dict(
    spin_mags = dict(
        mu_a = bilby.prior.Uniform(0, 1),
        sigma_a = bilby.prior.Uniform(0.005, 0.4),
    ),
    spin_tilts = dict(
        mu_cos = bilby.prior.Uniform(-1, 1),
        sigma_cos = bilby.prior.Uniform(0.1, 4),
        fcos = bilby.prior.Uniform(0, 1),
    ),
    redshift = dict(
        lamb = bilby.prior.Uniform(-10, 10),
    ),
    chi_eff = dict(
        mu_chi_eff = bilby.prior.Uniform(-1, 1),
        log_sigma_chi_eff = bilby.prior.Uniform(-2, 1),
    ),
)
