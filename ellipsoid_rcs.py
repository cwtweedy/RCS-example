"""
Monostatic RCS of a perfectly electrically conducting (PEC) ellipsoid.

The ellipsoid is centered at the origin with its major axis along y:

    (x / a_x)^2 + (y / a_y)^2 + (z / a_z)^2 = 1

By default the major axis is 6 cm long (a_y = 3 cm) and the minor axes are
3 cm long (a_x = a_z = 1.5 cm), i.e. a prolate spheroid. A plane wave is
incident from azimuth angle phi in the x-y plane (phi swept from -90 deg to
+90 deg) and the radar receives in the backscatter direction. phi = 0 looks
broadside at the ellipsoid (along x); phi = +/-90 deg looks end-on (along y).

Unlike a sphere, an ellipsoid has no simple exact series solution, so two
high-frequency approximations are computed:

  * Physical optics (PO): the surface current J = 2 n x H_inc is integrated
    numerically over the illuminated half of the surface. Captures
    frequency-dependent interference between the specular point and the
    shadow boundary.
  * Geometric optics (GO): the closed-form specular-point result
        sigma = pi a_x^2 a_y^2 a_z^2 / (a_x^2 cos^2 phi + a_y^2 sin^2 phi)^2
    which is frequency independent (valid as k * a_min -> infinity).

Both neglect creeping waves, so expect errors of order 1 dB when the
electrical size k * a_min is only a few (at 10 GHz, k * 1.5 cm ~ 3.1).
Run with --validate to compare PO with the exact Mie series for a sphere.

Usage examples:
    python ellipsoid_rcs.py                             # 6 cm x 3 cm, 10 GHz
    python ellipsoid_rcs.py --major 0.08 --minor 0.02
    python ellipsoid_rcs.py --freq 30e9 --save plot.png
    python ellipsoid_rcs.py --validate
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from sphere_rcs import C0, monostatic_rcs_vs_azimuth as mie_sphere_rcs


def go_monostatic_rcs(semi_axes, phi_deg):
    """Geometric-optics monostatic RCS (m^2) for incidence in the x-y plane."""
    ax_, ay, az = semi_axes
    phi = np.radians(phi_deg)
    denom = ax_ ** 2 * np.cos(phi) ** 2 + ay ** 2 * np.sin(phi) ** 2
    return np.pi * ax_ ** 2 * ay ** 2 * az ** 2 / denom ** 2


def po_monostatic_rcs(semi_axes, freq, phi_deg, samples_per_wavelength=20):
    """Physical-optics monostatic RCS (m^2) for incidence in the x-y plane.

    Monostatic PO for a PEC body reduces to (polarization independent)
        sigma = (4 pi / lambda^2) |  int_lit (n . r_hat) exp(2jk r_hat . r') dS  |^2
    where r_hat points from the body toward the radar.
    """
    ax_, ay, az = semi_axes
    k = 2.0 * np.pi * freq / C0
    wavelength = C0 / freq

    # Parameterize r(t, p) = (ax sin t cos p, ay sin t sin p, az cos t).
    # The round-trip phase 2k r_hat . r' changes by at most 2k a_max per radian,
    # so size the grid to resolve that with plenty of samples per cycle.
    a_max = max(semi_axes)
    n_t = max(200, int(np.ceil(samples_per_wavelength * 2.0 * a_max / wavelength * np.pi)))
    n_p = 2 * n_t
    t = (np.arange(n_t) + 0.5) * np.pi / n_t
    p = (np.arange(n_p) + 0.5) * 2.0 * np.pi / n_p
    dt, dp = np.pi / n_t, 2.0 * np.pi / n_p
    T, P = np.meshgrid(t, p, indexing="ij")
    st, ct, sp, cp = np.sin(T), np.cos(T), np.sin(P), np.cos(P)

    # Surface points and outward area-weighted normals N = (r_t x r_p) dt dp
    pts = np.stack([ax_ * st * cp, ay * st * sp, az * ct]).reshape(3, -1)
    nrm = np.stack([ay * az * st ** 2 * cp,
                    ax_ * az * st ** 2 * sp,
                    ax_ * ay * st * ct]).reshape(3, -1) * dt * dp

    phi = np.radians(phi_deg)
    rcs = np.empty_like(phi)
    for i, ph in enumerate(phi):
        r_hat = np.array([np.cos(ph), np.sin(ph), 0.0])
        proj = r_hat @ nrm                      # (n . r_hat) dS
        lit = proj > 0.0                        # illuminated half
        phase = 2.0 * k * (r_hat @ pts[:, lit])
        integral = np.sum(proj[lit] * np.exp(1j * phase))
        rcs[i] = 4.0 * np.pi / wavelength ** 2 * np.abs(integral) ** 2
    return rcs


def validate(freq):
    """Compare PO and GO against the exact Mie series for a 3 cm sphere."""
    a = 0.03
    phi = np.array([0.0, 37.0, 90.0])
    po = po_monostatic_rcs((a, a, a), freq, phi)
    go = go_monostatic_rcs((a, a, a), phi)
    mie = mie_sphere_rcs(a, freq, phi)
    print(f"Validation: PEC sphere a = {a * 100:.1f} cm, f = {freq / 1e9:.2f} GHz")
    print("  phi(deg)   Mie (dBsm)   PO (dBsm)   GO (dBsm)")
    for row in zip(phi, mie, po, go):
        print("  {:7.1f}   {:10.3f}  {:10.3f}  {:10.3f}".format(row[0], *(10 * np.log10(row[1:]))))


def main():
    parser = argparse.ArgumentParser(description="Monostatic RCS of a PEC ellipsoid vs azimuth.")
    parser.add_argument("--major", type=float, default=0.06,
                        help="major axis length (full, along y) in meters (default 0.06)")
    parser.add_argument("--minor", type=float, default=0.03,
                        help="minor axis length (full, along x and z) in meters (default 0.03)")
    parser.add_argument("--freq", type=float, default=10e9, help="frequency in Hz (default 10e9)")
    parser.add_argument("--points", type=int, default=361, help="number of azimuth samples (default 361)")
    parser.add_argument("--save", type=str, default=None, help="save plot to this file instead of showing it")
    parser.add_argument("--validate", action="store_true", help="compare PO with Mie for a sphere and exit")
    args = parser.parse_args()

    if args.validate:
        validate(args.freq)
        return

    semi_axes = (args.minor / 2.0, args.major / 2.0, args.minor / 2.0)  # (a_x, a_y, a_z)
    phi_deg = np.linspace(-90.0, 90.0, args.points)
    po_dbsm = 10.0 * np.log10(po_monostatic_rcs(semi_axes, args.freq, phi_deg))
    go_dbsm = 10.0 * np.log10(go_monostatic_rcs(semi_axes, phi_deg))

    wavelength = C0 / args.freq
    k = 2.0 * np.pi / wavelength
    print(f"Semi-axes (a_x, a_y, a_z) = ({semi_axes[0] * 100:.2f}, "
          f"{semi_axes[1] * 100:.2f}, {semi_axes[2] * 100:.2f}) cm")
    print(f"Frequency f = {args.freq / 1e9:.3f} GHz (lambda = {wavelength * 100:.3f} cm), "
          f"k a_min = {k * min(semi_axes):.2f}, k a_max = {k * max(semi_axes):.2f}")
    for ang in (0.0, 45.0, 90.0):
        i = np.argmin(np.abs(phi_deg - ang))
        print(f"  phi = {phi_deg[i]:5.1f} deg:  PO = {po_dbsm[i]:7.2f} dBsm,  GO = {go_dbsm[i]:7.2f} dBsm")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(phi_deg, po_dbsm, lw=2, label="Physical optics")
    ax.plot(phi_deg, go_dbsm, color="gray", ls="--", label="Geometric optics")
    ax.set_xlim(-90, 90)
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_xlabel("Azimuth angle of incidence, $\\phi$ (deg)")
    ax.set_ylabel("Monostatic RCS (dBsm)")
    ax.set_title(
        f"PEC ellipsoid monostatic RCS: {args.major * 100:.1f} cm x {args.minor * 100:.1f} cm "
        f"(major axis on y), f = {args.freq / 1e9:.2f} GHz"
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
