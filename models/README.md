# Model naming convention

-  `dataset`: If not specified, the analyses use GWTC-4 153 BBHs by default. Or for analyses end with `no231123`, it uses GWTC-4 152 BBHs excluding GW231123. There are analyses using 69 BBHs in GWTC-3 as well.
- `m2cut` infers the maximum m2 instead of a gap (this is used in analyses without GW231123 or only using GWTC-3 data).
- `m2gap` infers an empty gap in m2 distribution.
- `m2notch` infers the gap in m2 distribution by a notch filter (see Eq. 2 in [Fishbach et al.](https://iopscience.iop.org/article/10.3847/2041-8213/aba7b6)) with finite depth.
- `m2gap_identical` under `spin_transition` assumes the spin transition mass equal to the lower edge of the m2 gap.
- `m2gap_nonidentical` under `spin_transition` infers the spin transition mass and the lower edge of the m2 gap as different and independent parameters.
- `m2gap_fixed` under `spin_transition` uses a simple spin transition model as Eq. 5 in [Antonini et al.](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.134.011401).
- `m2gap_free` under `spin_transition` uses a flexible spin transition model as Eq. 6 in [Antonini et al.](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.134.011401).
- `pairing_function` model uses the mass model formalism following [Farah et al.](https://iopscience.iop.org/article/10.3847/1538-4357/ad0558). 