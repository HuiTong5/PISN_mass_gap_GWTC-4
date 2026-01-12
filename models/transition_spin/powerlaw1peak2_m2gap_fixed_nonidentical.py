import inspect
from gwpopulation.utils import xp, powerlaw, truncnorm
from gwpopulation.models.mass import double_power_law_primary_mass, truncnorm, BaseSmoothedMassDistribution
from gwpopulation.models.redshift import PowerLawRedshift
import bilby

def gap_powerlaw(xx, alpha, high1, low1, high2, low2):
    r"""
    Power-law probability

    .. math::
        p(x) = \frac{1 + \alpha}{x_\max^{1 + \alpha} - x_\min^{1 + \alpha}} x^\alpha

    Parameters
    ----------
    xx: float, array-like
        The abscissa values (:math:`x`)
    alpha: float, array-like
        The spectral index of the distribution (:math:`\alpha`)
    high: float, array-like
        The maximum of the distribution (:math:`x_\min`)
    low: float, array-like
        The minimum of the distribution (:math:`x_\max`)

    Returns
    -------
    prob: float, array-like
        The distribution evaluated at `xx`

    """
    norm1 = xp.where(
        xp.array(alpha) == -1,
        1 / xp.log(high1 / low1),
        (1 + alpha) / xp.array(high1 ** (1 + alpha) - low1 ** (1 + alpha)),
    )

    norm2 = xp.where(
        xp.array(alpha) == -1,
        1 / xp.log(high2 / low2),
        (1 + alpha) / xp.array(high2 ** (1 + alpha) - low2 ** (1 + alpha)),
    )

    prob = xp.power(xx, alpha)
    norm = 1/(1/norm1+1/norm2)
    prob *= norm
    prob *= ((xx <= high1) & (xx >= low1)) | ((xx <= high2) & (xx >= low2))
    return prob

def three_component_power_law_primary_mass(
    mass, alpha, mmin, mmax, lam_0, lam_1, mpp_1, sigpp_1, mpp_2, sigpp_2, gaussian_mass_maximum
    ):
    """
    A three-component double power law mass model.
    
    Parameters
    ----------
    mass: array-like
        The masses at which to evaluate the model (:math:`m`).
    alpha_1: float
        The power-law index below break (:math:`\alpha_1`).
    alpha_2: float
        The power-law index above break (:math:`\alpha_2`).
    mmin: float
        The minimum mass (:math:`m_{\min}`).
    mmax: float
        The maximum mass (:math:`m_{\max}`).
    break_mass: float
        The mass at which the break occurs (:math:`\delta`).
    lam_0: float
        The fraction of black holes in the power law (:math:`\hat{\lambda}_0`).
    lam_1: float
        The fraction of black holes in the lower Gaussian component (:math:`\hat{\lambda}_1`).
    mpp_1: float
        Mean of the lower mass Gaussian component (:math:`\mu_{m, 1}`).
    mpp_2: float
        Mean of the upper mass Gaussian component (:math:`\mu_{m, 2}`).
    sigpp_1: float
        Standard deviation of the lower mass Gaussian component (:math:`\sigma_{m, 1}`).
    sigpp_2: float
        Standard deviation of the upper mass Gaussian component (:math:`\sigma_{m, 2}`).
    gaussian_mass_maximum: float, optional
        Upper truncation limit of the Gaussian component. (default: 100)
        Note that this applies the same value to both.
    """
    lam_2 = 1 - lam_1 - lam_0
    p_pow = powerlaw(mass, alpha=-alpha, high=mmax, low=mmin)
    p_norm1 = truncnorm(
        mass, mu=mpp_1, sigma=sigpp_1, high=gaussian_mass_maximum, low=mmin
    )
    p_norm2 = truncnorm(
        mass, mu=mpp_2, sigma=sigpp_2, high=gaussian_mass_maximum, low=mmin
    )
    prob = lam_0 * p_pow +  lam_1 * p_norm1 + lam_2 * p_norm2
    return prob


class MultiPeakPowerLawSmoothedMassDistribution(BaseSmoothedMassDistribution):
    """
    Broken power law mass distribution with two Gaussian components with smoothing.


    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
    alpha_1: float
        Power law exponent of the primary mass distribution below the break.
    alpha_2: float
        Power law exponent of the primary mass distribution above the break.
    beta: float
        Power law exponent of the mass ratio distribution.
    mmin_1: float
        Minimum primary black hole mass.
    mmin_2: float
        Minimum secondary black hole mass.
    mmax: float
        Maximum mass in the powerlaw distributed component.
    break_mass: float
        Mass at which the power law transitions from alpha_1 to alpha_2.
    lam_0: float
        Fraction of black holes in the power law component.
    lam_1: float
        Fraction of black holes in the lower mass Gaussian component.
    mpp_1: float
        Mean of the lower mass Gaussian component.
    mpp_2: float
        Mean of the higher mass Gaussian component.
    sigpp_1: float
        Standard deviation of the lower mass Gaussian component.
    sigpp_2: float
        Standard deviation of the higher mass Gaussian component.
    delta_m_1: float
        Rise length of the low end of the primary mass distribution.
    delta_m_2: float
        Rise length of the secondary mass distribution.

    Notes
    -----
    The Gaussian components are bounded between [`mmin`, `self.mmax`].
    This means that the `mmax` parameter is _not_ the global maximum.
    """

    primary_model = three_component_power_law_primary_mass

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)
    
    def __init__(self, mmin=2, mmax=300, normalization_shape=(2000, 4000), cache=True, spacing="log"):
        self.mmin = mmin
        self.mmax = mmax
        if spacing == "log":
            self.m1s = xp.logspace(xp.log10(mmin), xp.log10(mmax), normalization_shape[0])
        elif spacing == "linear":
            self.m1s = xp.linspace(mmin, mmax, normalization_shape[0])
        self.qs = xp.linspace(0.001, 1, normalization_shape[1])
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.cache = cache
        self.spacing = spacing

    def __call__(self, dataset, *args, **kwargs):
        beta = kwargs.pop("beta")
        mmin_1 = kwargs.pop("mlow_1", self.mmin)
        mmin_2 = kwargs.pop("mlow_2", self.mmin)
        delta_m_1 = kwargs.pop("delta_m_1", 0)
        delta_m_2 = kwargs.pop("delta_m_2", 0)
        mmax = kwargs.get("mmax", self.mmax)
        if "jax" not in xp.__name__:
            if mmin_1 < self.mmin or mmin_2 < self.mmin:
                raise ValueError(
                    "{self.__class__}: mlow ({mmin}) < self.mmin ({self.mmin})"
                )
            if mmax > self.mmax:
                raise ValueError(
                    "{self.__class__}: mmax ({mmax}) > self.mmax ({self.mmax})"
                )
        p_m1 = self.p_m1(dataset, mmin=mmin_1, delta_m=delta_m_1, **kwargs, **self.kwargs)
        p_q = self.p_q(dataset, beta=beta, mmin=mmin_2, delta_m=delta_m_2)
        prob = p_m1 * p_q
        return prob

    def _cache_q_norms(self, masses):
        """
        Cache the information necessary for linear interpolation of the mass
        ratio normalisation
        """
        from gwpopulation.models.interped import _setup_interpolant

        if self.spacing == "log":   
            func = xp.log
        else:
            func = xp.array

        self._q_interpolant = _setup_interpolant(
            func(self.m1s), func(masses), kind="linear", backend=xp
        )
        
    @property
    def variable_names(self):
        vars = getattr(
            self.primary_model,
            "variable_names",
            inspect.getfullargspec(self.primary_model).args[1:],
        )
        vars += ["beta", "delta_m_1", "delta_m_2", "mlow_1", "mlow_2"]
        vars.remove("mmin")
        vars = set(vars).difference(self.kwargs.keys())
        return vars

class m2gap_MultiPeakPowerLawSmoothedMassDistribution(MultiPeakPowerLawSmoothedMassDistribution):

    def __init__(self, mmin=2, mmax=300, normalization_shape=(2000, 4000), cache=True, spacing="log"):
        self.mmin = mmin
        self.mmax = mmax
        if spacing == "log":
            self.m1s = xp.logspace(xp.log10(mmin), xp.log10(mmax), normalization_shape[0])
        elif spacing == "linear":
            self.m1s = xp.linspace(mmin, mmax, normalization_shape[0])
        self.qs = xp.linspace(0.001, 1, normalization_shape[1])
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.cache = cache
        self.spacing = spacing

    def __call__(self, dataset, *args, **kwargs):
        beta = kwargs.pop("beta")
        mmin_1 = kwargs.pop("mlow_1", self.mmin)
        mmin_2 = kwargs.pop("mlow_2", self.mmin)
        delta_m_1 = kwargs.pop("delta_m_1", 0)
        delta_m_2 = kwargs.pop("delta_m_2", 0)
        mmax = kwargs.get("mmax", self.mmax)
        gap_low = kwargs.pop("gap_low")
        gap_width = kwargs.pop("gap_width")

        if "jax" not in xp.__name__:
            if mmin_1 < self.mmin or mmin_2 < self.mmin:
                raise ValueError(
                    "{self.__class__}: mlow ({mmin}) < self.mmin ({self.mmin})"
                )
            if mmax > self.mmax:
                raise ValueError(
                    "{self.__class__}: mmax ({mmax}) > self.mmax ({self.mmax})"
                )
        p_m1 = self.p_m1(dataset, mmin=mmin_1, delta_m=delta_m_1, **kwargs, **self.kwargs)
        p_q = self.p_q(dataset, beta=beta, mmin=mmin_2, delta_m=delta_m_2, gap_low=gap_low, gap_width=gap_width)
        prob = p_m1 * p_q
        return prob

    def p_q(self, dataset, beta, mmin, delta_m, gap_low, gap_width):
        p_q = 0.0
        p_q += (dataset['mass_1'] <= gap_low) * powerlaw(dataset['mass_ratio'], beta, 1, mmin / dataset['mass_1']) # for case where mass_1 is lower than the lower boundary of mass gap
        p_q += (dataset['mass_1'] > gap_low) * (dataset['mass_1'] <= gap_low + gap_width) * powerlaw(dataset['mass_ratio'], beta, gap_low / dataset['mass_1'], mmin / dataset['mass_1'])
        p_q += (dataset['mass_1'] > gap_low + gap_width) * \
                gap_powerlaw(dataset['mass_ratio'], beta, gap_low/dataset['mass_1'], mmin / dataset['mass_1'], 1, (gap_low+gap_width)/dataset['mass_1'])
        p_q *= self.smoothing(
            dataset["mass_1"] * dataset["mass_ratio"],
            mmin=mmin,
            mmax=dataset["mass_1"],
            delta_m=delta_m,
        )
        try:
            if self.cache:
                p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m, gap_low=gap_low, gap_width=gap_width)
            else:
                self._cache_q_norms(dataset["mass_1"])
                p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m, gap_low=gap_low, gap_width=gap_width)
        except (AttributeError, TypeError, ValueError):
            self._cache_q_norms(dataset["mass_1"])
            p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m, gap_low=gap_low, gap_width=gap_width)

        return xp.nan_to_num(p_q)

    def norm_p_q(self, beta, mmin, delta_m, gap_low, gap_width):
        """Calculate the mass ratio normalisation by linear interpolation"""
        p_q = 0.0
        p_q += (self.m1s_grid <= gap_low) * powerlaw(self.qs_grid, beta, 1, mmin / self.m1s_grid) # for case where mass_1 is lower than the lower boundary of mass gap
        p_q += (self.m1s_grid > gap_low) * (self.m1s_grid <= gap_low + gap_width) * powerlaw(self.qs_grid, beta, gap_low / self.m1s_grid, mmin / self.m1s_grid)
        p_q += (self.m1s_grid > gap_low + gap_width) * \
                gap_powerlaw(self.qs_grid, beta, gap_low/self.m1s_grid, mmin / self.m1s_grid, 1, (gap_low+gap_width)/self.m1s_grid)

        p_q *= self.smoothing(
            self.m1s_grid * self.qs_grid, mmin=mmin, mmax=self.m1s_grid, delta_m=delta_m
        )

        norms = xp.nan_to_num(xp.trapz(p_q, self.qs, axis=0)) * (delta_m != 0) + 1 * (
            delta_m == 0
        )

        return self._q_interpolant(norms)


    @property
    def variable_names(self):
        vars = getattr(
            self.primary_model,
            "variable_names",
            inspect.getfullargspec(self.primary_model).args[1:],
        )
        vars += ["beta", "delta_m_1", "delta_m_2", "mlow_1", "mlow_2", "gap_low", "gap_width"]
        vars.remove("mmin")
        vars = set(vars).difference(self.kwargs.keys())
        return vars

def mlow_2_condition(reference_params, mlow_1):
    return dict(
        minimum=reference_params["minimum"],
        maximum=mlow_1
        )

class Transition_chi_eff:
    @property
    def variable_names(self):
        vars = ["m_t", "w", "log_sigma_chi_eff_low", "mu_chi_eff_low"]
        return vars

    def __call__(self, dataset, *args, **kwargs):
        m_t = kwargs['m_t']
        w = kwargs['w']
        log_sigma_chi_eff_low = kwargs['log_sigma_chi_eff_low']
        mu_chi_eff_low = kwargs['mu_chi_eff_low']

        p_chi = (dataset['mass_1']<m_t) * truncnorm(dataset['chi_eff'], mu_chi_eff_low, 10**(log_sigma_chi_eff_low), 1, -1)
        p_chi += (dataset['mass_1']>=m_t) * self.p_Uniform_chi_eff(dataset['chi_eff'], w)

        return p_chi
    
    def p_Uniform_chi_eff(self, chi_eff, width):

        p = ((chi_eff >= -width) * (chi_eff <= width)) * 1/(2*width)

        return p

models = dict(
    mass = m2gap_MultiPeakPowerLawSmoothedMassDistribution,
    chi_eff = Transition_chi_eff,
)


priors = dict(
    mass = dict(
        alpha = bilby.prior.Uniform(-4, 12),
        beta = bilby.prior.Uniform(-2, 7),
        mmax = 300,
        mlow_1 = bilby.prior.Uniform(3, 10),
        mlow_2 = bilby.prior.ConditionalUniform(minimum=3, maximum=10, condition_func=mlow_2_condition),
        mpp_1 = bilby.prior.Uniform(minimum=5, maximum=20, name='mpp_1', latex_label='$\\mu_{m1}$'),
        mpp_2 = bilby.prior.Uniform(minimum=25, maximum=60, name='mpp_2', latex_label='$\\mu_{m2}$'),
        sigpp_1 = bilby.prior.Uniform(minimum=0, maximum=10, name='sigpp_1', latex_label='$\\sigma_{m1}$'),
        sigpp_2 = bilby.prior.Uniform(minimum=0, maximum=10, name='sigpp_2', latex_label='$\\sigma_{m2}$'),
        delta_m_1 = bilby.prior.Uniform(minimum=0, maximum=10, name='delta_m_1', latex_label='$\\delta_{m1}$', boundary='reflective'),
        delta_m_2 = bilby.prior.Uniform(minimum=0, maximum=10, name='delta_m_2', latex_label='$\\delta_{m2}$', boundary='reflective'),
        lam_0 = bilby.prior.DirichletElement(order=0, n_dimensions=3, label='lam_'),
        lam_1 = bilby.prior.DirichletElement(order=1, n_dimensions=3, label='lam_'),
        gap_low = bilby.prior.Uniform(20, 150),
        gap_width = bilby.prior.Uniform(0, 150),
        H0 = 67.90,
        Om0 = 0.3065,
    ),
    chi_eff = dict(
        m_t = bilby.prior.Uniform(20, maximum=100),
        w = 0.47,
        log_sigma_chi_eff_low = bilby.prior.Uniform(minimum=-2, maximum=1),
        mu_chi_eff_low = bilby.prior.Uniform(minimum=-1, maximum=1),
    )
)
