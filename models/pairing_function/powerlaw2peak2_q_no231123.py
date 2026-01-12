import jax
import jax.numpy as jnp
from gwpopulation.utils import powerlaw, truncnorm
from gwpopulation.models.redshift import PowerLawRedshift
import bilby

xp = jnp

def smoothing(masses, mmin, mmax, delta_m):
    shifted_mass = jnp.nan_to_num((masses - mmin) / delta_m, nan=0)
    shifted_mass = jnp.clip(shifted_mass, 1e-6, 1 - 1e-6)
    exponent = 1 / shifted_mass - 1 / (1 - shifted_mass)
    window = jax.scipy.special.expit(-exponent)
    window *= (masses >= mmin) * (masses <= mmax)
    return window

def mass_1_model(
    dataset,
    alpha1_m1, alpha2_m1, mb_m1, mmin_m1, mmax_m1, delta_m1, 
    mu1_m1, mu2_m1, sigma1_m1, sigma2_m1, lam_m1_0, lam_m1_1
):
    lam_m1_2 = 1 - lam_m1_0 - lam_m1_1
    def shape(m):
        pl = (
            (mmin_m1 <= m) * (m < mb_m1) * (m / mb_m1)**alpha1_m1
            + (mb_m1 <= m) * (m / mb_m1)**alpha2_m1
        ) / (
            (mb_m1**(alpha1_m1+1) - mmin_m1**(alpha1_m1+1)) / (alpha1_m1+1) / mb_m1**alpha1_m1
            + (mmax_m1**(alpha2_m1+1) - mb_m1**(alpha2_m1+1)) / (alpha2_m1+1) / mb_m1**alpha2_m1
        )
        n1 = truncnorm(m, mu1_m1, sigma1_m1, mmax_m1, mmin_m1)
        n2 = truncnorm(m, mu2_m1, sigma2_m1, mmax_m1, mmin_m1)
        p = lam_m1_0 * pl + lam_m1_1 * n1 + lam_m1_2 * n2

        lo = smoothing(m, mmin_m1, mmax_m1, delta_m1)

        return p * lo

    x = jnp.linspace(mmin_m1, mmax_m1, 10_000)
    norm = jnp.trapezoid(shape(x), x)

    return shape(dataset['mass_1']) / norm

def mass_2_model(
    dataset,
    alpha1_m2, alpha2_m2, mb_m2, mmin_m2, mmax_m2, delta_m2, 
    mu1_m2, mu2_m2, sigma1_m2, sigma2_m2, lam_m2_0, lam_m2_1
):
    lam_m2_2 = 1 - lam_m2_0 - lam_m2_1
    def shape(m):
        pl = (
            (mmin_m2 <= m) * (m < mb_m2) * (m / mb_m2)**alpha1_m2
            + (mb_m2 <= m) * (m / mb_m2)**alpha2_m2
        ) / (
            (mb_m2**(alpha1_m2+1) - mmin_m2**(alpha1_m2+1)) / (alpha1_m2+1) / mb_m2**alpha1_m2
            + (mmax_m2**(alpha2_m2+1) - mb_m2**(alpha2_m2+1)) / (alpha2_m2+1) / mb_m2**alpha2_m2
        )
        n1 = truncnorm(m, mu1_m2, sigma1_m2, mmax_m2, mmin_m2)
        n2 = truncnorm(m, mu2_m2, sigma2_m2, mmax_m2, mmin_m2)
        p = lam_m2_0 * pl + lam_m2_1 * n1 + lam_m2_2 * n2

        lo = smoothing(m, mmin_m2, mmax_m2, delta_m2)

        return p * lo

    x = jnp.linspace(mmin_m2, mmax_m2, 10_000)
    norm = jnp.trapezoid(shape(x), x)

    if 'mass_2' in dataset:
        return shape(dataset['mass_2']) / norm
    return shape(dataset['mass_1']*dataset['mass_ratio']) / norm

def mass_ratio_model(dataset, beta, alpha1_m2, alpha2_m2, mb_m2, mmin_m2, mmax_m2, delta_m2, 
    mu1_m2, mu2_m2, sigma1_m2, sigma2_m2, lam_m2_0, lam_m2_1):

    p_m2 = mass_2_model(dataset, alpha1_m2, alpha2_m2, mb_m2, mmin_m2, mmax_m2, delta_m2, mu1_m2, mu2_m2, sigma1_m2, sigma2_m2, lam_m2_0, lam_m2_1)

    return p_m2 * powerlaw(dataset['mass_ratio'], beta, 1, 3/300) * dataset['mass_1']

def mmin_m2_condition(reference_params, mmin_m1):
    return dict(
        minimum=reference_params["minimum"],
        maximum=mmin_m1
        )

def mmax_m2_condition(reference_params, mmax_m1):
    return dict(
        minimum=reference_params["minimum"],
        maximum=mmax_m1
        )

models = dict(
    mass_1 = mass_1_model,
    mass_ratio = mass_ratio_model,
)


priors = dict(
    mass_1 = dict(
        alpha1_m1 = bilby.prior.Uniform(-12, 4),
        alpha2_m1 = bilby.prior.Uniform(-12, 4),
        mb_m1 = bilby.prior.Uniform(20, 50),
        mmin_m1 = bilby.prior.Uniform(3, 10),
        mmax_m1 = bilby.prior.Uniform(20, 200),
        delta_m1 = bilby.prior.Uniform(0, 10),
        mu1_m1 = bilby.prior.Uniform(5, 20),
        mu2_m1 = bilby.prior.Uniform(25, 60),
        sigma1_m1 = bilby.prior.Uniform(0, 10),
        sigma2_m1 = bilby.prior.Uniform(0, 10),
        lam_m1_0 = bilby.prior.DirichletElement(order=0, n_dimensions=3, label='lam_m1_'),
        lam_m1_1 = bilby.prior.DirichletElement(order=1, n_dimensions=3, label='lam_m1_'),
    ),
    mass_ratio = dict(
        alpha1_m2 = bilby.prior.Uniform(-12, 4),
        alpha2_m2 = bilby.prior.Uniform(-12, 4),
        mb_m2 = bilby.prior.Uniform(20, 50),
        mmin_m2 = bilby.prior.ConditionalUniform(minimum=3, maximum=10, condition_func=mmin_m2_condition),
        mmax_m2 = bilby.prior.ConditionalUniform(minimum=20, maximum=200, condition_func=mmax_m2_condition),
        delta_m2 = bilby.prior.Uniform(0, 10),
        mu1_m2 = bilby.prior.Uniform(5, 20),
        mu2_m2 = bilby.prior.Uniform(25, 60),
        sigma1_m2 = bilby.prior.Uniform(0, 10),
        sigma2_m2 = bilby.prior.Uniform(0, 10),
        lam_m2_0 = bilby.prior.DirichletElement(order=0, n_dimensions=3, label='lam_m2_'),
        lam_m2_1 = bilby.prior.DirichletElement(order=1, n_dimensions=3, label='lam_m2_'),
        beta = bilby.prior.Uniform(-4, 12),
        H0 = 67.90,
        Om0 = 0.3065,
    ),
)
