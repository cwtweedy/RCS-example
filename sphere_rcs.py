"""
Monostatic RCS of a perfectly electrically conducting (PEC) sphere.

The sphere sits at the origin. A plane wave is incident from azimuth angle
phi in the x-y plane (phi swept from -90 deg to +90 deg), and the radar
receives in the backscatter direction (monostatic). The scattered field is
computed with the exact Mie series solution.

Because a sphere is rotationally symmetric, the monostatic RCS is the same
for every incidence direction -- the plot is a flat line, which is the
expected (and correct) answer. The program still computes the geometry
explicitly at each angle so it can be adapted to bistatic cases.

Usage examples:
    python sphere_rcs.py                       # a = 3 cm, f = 10 GHz
    python sphere_rcs.py --radius 0.05         # a = 5 cm
    python sphere_rcs.py --radius 0.03 --freq 3e9
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn

C0 = 299_792_458.0  # speed of light in vacuum (m/s)


def mie_coefficients_pec(x):
    """Mie coefficients a_n, b_n for a PEC sphere with size parameter x = k*a.

    Bohren & Huffman convention, in the limit of infinite conductivity:
        a_n = psi_n'(x) / xi_n'(x),   b_n = psi_n(x) / xi_n(x)
    where psi_n(x) = x j_n(x) and xi_n(x) = x h_n^(1)(x) are Riccati-Bessel
    functions.
    """
    # Wiscombe's criterion for the number of terms needed for convergence
    n_max = int(np.ceil(x + 4.0 * x ** (1.0 / 3.0) + 2.0))
    n = np.arange(1, n_max + 1)

    jn = spherical_jn(n, x)
    jn_p = spherical_jn(n, x, derivative=True)
    yn = spherical_yn(n, x)
    yn_p = spherical_yn(n, x, derivative=True)

    hn = jn + 1j * yn
    hn_p = jn_p + 1j * yn_p

    psi = x * jn
    psi_p = jn + x * jn_p
    xi = x * hn
    xi_p = hn + x * hn_p

    a_n = psi_p / xi_p
    b_n = psi / xi
    return n, a_n, b_n


def scattering_amplitudes(n, a_n, b_n, theta_s):
    """Far-field amplitudes S1(theta_s), S2(theta_s) for scattering angle theta_s."""
    mu = np.cos(theta_s)
    pi_n = np.zeros(len(n) + 1)  # pi_n[0] = 0
    tau_n = np.zeros(len(n) + 1)
    pi_n[1] = 1.0
    tau_n[1] = mu
    for m in range(2, len(n) + 1):
        pi_n[m] = ((2 * m - 1) * mu * pi_n[m - 1] - m * pi_n[m - 2]) / (m - 1)
        tau_n[m] = m * mu * pi_n[m] - (m + 1) * pi_n[m - 1]

    weights = (2 * n + 1) / (n * (n + 1))
    s1 = np.sum(weights * (a_n * pi_n[1:] + b_n * tau_n[1:]))
    s2 = np.sum(weights * (a_n * tau_n[1:] + b_n * pi_n[1:]))
    return s1, s2


def monostatic_rcs_vs_azimuth(radius, freq, phi_deg):
    """Monostatic RCS (m^2) of a PEC sphere at the origin for each azimuth angle.

    The incident wave propagates toward the origin from direction phi in the
    x-y plane; the receiver is co-located with the transmitter.
    """
    k = 2.0 * np.pi * freq / C0
    n, a_n, b_n = mie_coefficients_pec(k * radius)

    phi = np.radians(phi_deg)
    rcs = np.empty_like(phi)
    for i, p in enumerate(phi):
        # Radar position direction and incident propagation direction
        r_radar = np.array([np.cos(p), np.sin(p), 0.0])
        k_inc = -r_radar
        # Scattered wave travels back toward the radar
        k_scat = r_radar
        theta_s = np.arccos(np.clip(np.dot(k_inc, k_scat), -1.0, 1.0))

        s1, _ = scattering_amplitudes(n, a_n, b_n, theta_s)
        # At backscatter |S1| = |S2|, so co-pol RCS is polarization independent
        rcs[i] = 4.0 * np.pi * np.abs(s1) ** 2 / k ** 2
    return rcs


def main():
    parser = argparse.ArgumentParser(description="Monostatic RCS of a PEC sphere vs azimuth.")
    parser.add_argument("--radius", type=float, default=0.03, help="sphere radius in meters (default 0.03)")
    parser.add_argument("--freq", type=float, default=10e9, help="frequency in Hz (default 10e9)")
    parser.add_argument("--points", type=int, default=361, help="number of azimuth samples (default 361)")
    parser.add_argument("--save", type=str, default=None, help="save plot to this file instead of showing it")
    args = parser.parse_args()

    phi_deg = np.linspace(-90.0, 90.0, args.points)
    rcs = monostatic_rcs_vs_azimuth(args.radius, args.freq, phi_deg)
    rcs_dbsm = 10.0 * np.log10(rcs)

    wavelength = C0 / args.freq
    ka = 2.0 * np.pi * args.radius / wavelength
    optical = np.pi * args.radius ** 2

    print(f"Radius a        = {args.radius * 100:.3f} cm")
    print(f"Frequency f     = {args.freq / 1e9:.3f} GHz (lambda = {wavelength * 100:.3f} cm)")
    print(f"ka              = {ka:.4f}")
    print(f"Monostatic RCS  = {rcs.mean():.6e} m^2 ({rcs_dbsm.mean():.3f} dBsm)")
    print(f"sigma / (pi a^2) = {rcs.mean() / optical:.4f}")
    print(f"Variation over azimuth = {rcs_dbsm.max() - rcs_dbsm.min():.2e} dB")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(phi_deg, rcs_dbsm, lw=2, label="Mie series (exact)")
    ax.axhline(10.0 * np.log10(optical), color="gray", ls="--", label=r"Optical limit $\pi a^2$")
    ax.set_xlim(-90, 90)
    ax.set_xticks(np.arange(-90, 91, 30))
    center = rcs_dbsm.mean()
    ax.set_ylim(center - 10, center + 10)
    ax.set_xlabel("Azimuth angle of incidence, $\\phi$ (deg)")
    ax.set_ylabel("Monostatic RCS (dBsm)")
    ax.set_title(
        f"PEC sphere monostatic RCS: a = {args.radius * 100:.2f} cm, "
        f"f = {args.freq / 1e9:.2f} GHz (ka = {ka:.2f})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Plot saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
