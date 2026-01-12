## Default Population Model

Below is a description of the default population model chosen by the paper writing team based on the preliminary results submitted by analysts.
### Primary mass:

2 Power Law + 2 Peaks (model proposed in https://arxiv.org/abs/2302.07289)

$$
p(m_1) = f_{p,1} N(m_1|\mu_{m,1},\sigma_{m,1}) + f_{p,2} N(m_2|\mu_{m,2},\sigma_{m,2}) + (1-f_{p,1}-f_{p,2}) \Gamma(m_1),
$$

This is the superposition of three components: two Gaussians and a tapered, broken power law $\Gamma(m_1)$:

$$
\Gamma(m_1) \propto \begin{cases}e^{-\frac{(m_1-m_\mathrm{min})^2}{2\delta m_\mathrm{min}^2}} \left(\dfrac{m_1}{m_b}\right)^{\alpha_1} & (m_1<m_\mathrm{min}) \\ \left(\dfrac{m_1}{m_b}\right)^{\alpha_1} & (m_\mathrm{min} \leq m_1 < m_b) \\[8pt] \left(\dfrac{m_1}{m_b}\right)^{\alpha_2} & (m_b \leq m_1 < m_\mathrm{max}) \\[8pt] e^{-\frac{(m_1-m_\mathrm{max})^2}{2\delta m_\mathrm{max}^2}} \left(\dfrac{m_1}{m_b}\right)^{\alpha_2} & (\mathrm{else}),\end{cases}
$$

### Mass ratio:

Power Law

$$
p(q) = q^\beta
$$
### Spin Magnitude

Gaussian defined between [0,1]. Not assumed to be independently and identically distributed.
$$
p(a_i) = \mathcal{N}(\mu_{a_i},\sigma_{a_i})
$$
### Cosine Tilt

Mixture between a Gaussian and an isotropic distribution.

$$
p(\text{cos}\theta_i) = \delta/2 + (1-\delta)\mathcal{N}(\mu_{\theta_i}, \sigma_{\theta_i})
$$
### Redshift
Power law in merger rate density

$$
p(z) = \frac{dVc}{dz} (1+z)^{\lambda-1}
$$