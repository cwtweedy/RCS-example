"""
Monostatic RCS of a perfectly electrically conducting (PEC) flat plate.

The plate is infinitely thin, centered at the origin, and lies in the y-z
plane (normal along x):

    x = 0,  |y| <= L_y / 2,  |z| <= L_z / 2

By default it is 12 cm x 12 cm. A plane wave is incident from azimuth angle
phi in the x-y plane (phi swept from -90 deg to +90 deg) and the radar
receives in the backscatter direction. phi = 0 is normal incidence on the
plate; phi = +/-90 deg is edge-on (grazing). The sweep rotates the radar
about the z-axis, so the two edges parallel to z (at y = +/- L_y / 2) set
the lobe structure.

Two models are computed:

  * Physical optics (PO): the surface current J = 2 n x H_inc is integrated
    numerically over the illuminated face. For incidence in the x-y plane
    this also has the closed form
        sigma = (4 pi A^2 / lambda^2) cos^2(phi) [sin(X) / X]^2,
        X = k L_y sin(phi),  A = L_y L_z
    PO ignores edge diffraction, so it is polarization independent and
    drops to zero at grazing, where the true RCS is finite.

  * Edge-diffraction model: the plate's cross-section in the x-y plane is a
    strip of width L_y. Its 2D echo width sigma_2D is solved exactly
    (to discretization accuracy) with a method-of-moments (MoM) solution of
    the electric-field integral equation, which includes all diffraction
    from the z-parallel edges (single, multiple and grazing-incidence).
    It is converted to 3D with the standard long-body scaling
        sigma_3D = sigma_2D * 2 L_z^2 / lambda
    (exact for PO at broadside; it neglects diffraction from the y-parallel
    edges and corners, which is weak in this principal-plane cut).
    Two polarizations are computed:
        VV: E parallel to z (parallel to the diffracting edges)
        HH: E in the x-y plane (perpendicular to the diffracting edges)

Run with --validate to check the MoM against PO at broadside, against
itself at a finer mesh, and against Keller's geometrical theory of
diffraction (GTD) for the strip at intermediate angles.

Usage examples:
    python plate_rcs.py                          # 12 cm x 12 cm, 10 GHz
    python plate_rcs.py --ly 0.20 --lz 0.10
    python plate_rcs.py --curves po,hh --save plot.png
    python plate_rcs.py --validate
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2

from sphere_rcs import C0

ETA0 = 376.730313668  # free-space wave impedance (ohms)

# Gauss-Legendre rule for the smooth part of the segment integrals
_GL_X, _GL_W = np.polynomial.legendre.leggauss(10)


def po_monostatic_rcs(ly, lz, freq, phi_deg, samples_per_wavelength=20):
    """Physical-optics monostatic RCS (m^2) for incidence in the x-y plane.

    Monostatic PO for a PEC body reduces to (polarization independent)
        sigma = (4 pi / lambda^2) |  int_lit (n . r_hat) exp(2jk r_hat . r') dS  |^2
    where r_hat points from the body toward the radar.
    """
    k = 2.0 * np.pi * freq / C0
    wavelength = C0 / freq

    # The round-trip phase 2k r_hat . r' needs sampling at a fraction of lambda/2
    step = wavelength / (2.0 * samples_per_wavelength)
    n_y = max(50, int(np.ceil(ly / step)))
    n_z = max(50, int(np.ceil(lz / step)))
    y = (np.arange(n_y) + 0.5) * ly / n_y - ly / 2.0
    z = (np.arange(n_z) + 0.5) * lz / n_z - lz / 2.0
    dA = (ly / n_y) * (lz / n_z)
    Y, Z = np.meshgrid(y, z, indexing="ij")
    pts = np.stack([np.zeros_like(Y), Y, Z]).reshape(3, -1)

    phi = np.radians(phi_deg)
    rcs = np.empty_like(phi)
    for i, ph in enumerate(phi):
        r_hat = np.array([np.cos(ph), np.sin(ph), 0.0])
        # The face whose normal points toward the radar is the illuminated one
        n_hat = np.array([1.0, 0.0, 0.0]) if r_hat[0] >= 0.0 else np.array([-1.0, 0.0, 0.0])
        proj = (n_hat @ r_hat) * dA
        phase = 2.0 * k * (r_hat @ pts)
        integral = proj * np.sum(np.exp(1j * phase))
        rcs[i] = 4.0 * np.pi / wavelength ** 2 * np.abs(integral) ** 2
    return rcs


def po_closed_form_rcs(ly, lz, freq, phi_deg):
    """Closed-form PO monostatic RCS (m^2) for incidence in the x-y plane."""
    wavelength = C0 / freq
    k = 2.0 * np.pi / wavelength
    phi = np.radians(phi_deg)
    area = ly * lz
    x = k * ly * np.sin(phi)
    sinc = np.sinc(x / np.pi)  # numpy sinc is sin(pi u) / (pi u)
    return 4.0 * np.pi * area ** 2 / wavelength ** 2 * np.cos(phi) ** 2 * sinc ** 2


def _hankel_segment_integrals(k, y_obs, a, b):
    """Matrix of int_a^b H0^(2)(k |y - y'|) dy' for every observation point y.

    The log singularity -j (2/pi) ln|u| is integrated analytically; the
    bounded remainder is integrated with Gauss-Legendre quadrature.
    """
    y = np.asarray(y_obs)[:, None]
    a = np.asarray(a)[None, :]
    b = np.asarray(b)[None, :]

    # Analytic part: int ln|y' - y| dy' with antiderivative u ln|u| - u
    def antideriv(u):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(u == 0.0, 0.0, u * np.log(np.abs(u)) - u)

    ln_int = antideriv(b - y) - antideriv(a - y)
    singular = -1j * (2.0 / np.pi) * (ln_int + (b - a) * np.log(k))

    # Remainder H0(k|u|) + j (2/pi) ln(k|u|) is continuous
    mid, half = (a + b) / 2.0, (b - a) / 2.0
    yq = mid[..., None] + half[..., None] * _GL_X           # (1, N, Q)
    x = k * np.abs(yq - y[..., None])                        # (M, N, Q)
    rem = hankel2(0, x) + 1j * (2.0 / np.pi) * np.log(x)
    smooth = np.sum(rem * _GL_W, axis=-1) * half
    return singular + smooth


def strip_echo_width_mom(width, freq, phi_deg, segments_per_wavelength=40):
    """2D monostatic echo width (m) of a PEC strip, both polarizations, via MoM.

    The strip occupies x = 0, |y| <= width / 2 and is illuminated from
    azimuth phi in the x-y plane (phi = 0 is broadside). Returns
    (sigma_vv, sigma_hh) for E parallel to the strip edges (z) and
    E in the x-y plane.

    VV uses pulse basis functions with point matching. HH uses triangle
    (rooftop) basis functions, which force the current to zero at the
    edges, with the mixed-potential EFIE and pulse testing.
    """
    wavelength = C0 / freq
    k = 2.0 * np.pi / wavelength
    n_seg = max(20, int(np.ceil(segments_per_wavelength * width / wavelength)))
    d = width / n_seg
    nodes = np.linspace(-width / 2.0, width / 2.0, n_seg + 1)
    phi = np.radians(np.atleast_1d(phi_deg))
    s, c = np.sin(phi), np.cos(phi)

    # --- VV: E_z^inc(c_m) = (k eta / 4) sum_n I_n int_seg_n H0 dy'
    centers = (nodes[:-1] + nodes[1:]) / 2.0
    z_vv = (k * ETA0 / 4.0) * _hankel_segment_integrals(k, centers, nodes[:-1], nodes[1:])
    v_vv = np.exp(1j * k * np.outer(centers, s))
    i_vv = np.linalg.solve(z_vv, v_vv)
    f_vv = d * np.sum(i_vv * np.exp(1j * k * np.outer(centers, s)), axis=0)
    sigma_vv = k * ETA0 ** 2 / 4.0 * np.abs(f_vv) ** 2

    # --- HH: d E_y^inc(y_m) = j w A_y(y_m) d + Phi(y_m + d/2) - Phi(y_m - d/2)
    y_n = nodes[1:-1]                      # interior nodes carry the rooftops
    a_term = _hankel_segment_integrals(k, y_n, y_n - d / 2.0, y_n + d / 2.0)

    def potential(y_obs):
        left = _hankel_segment_integrals(k, y_obs, nodes[:-2], nodes[1:-1])
        right = _hankel_segment_integrals(k, y_obs, nodes[1:-1], nodes[2:])
        return ETA0 / (4.0 * k * d) * (left - right)

    z_hh = (k * ETA0 / 4.0) * d * a_term + potential(y_n + d / 2.0) - potential(y_n - d / 2.0)
    v_hh = d * (-c) * np.exp(1j * k * np.outer(y_n, s))
    i_hh = np.linalg.solve(z_hh, v_hh)
    f_hh = d * c * np.sum(i_hh * np.exp(1j * k * np.outer(y_n, s)), axis=0)
    sigma_hh = k * ETA0 ** 2 / 4.0 * np.abs(f_hh) ** 2
    return sigma_vv, sigma_hh


def strip_echo_width_gtd(width, freq, phi_deg):
    """2D monostatic echo width (m) of a PEC strip from Keller's GTD.

    Single diffraction from the two edges with the half-plane (n = 2)
    diffraction coefficient. Singular at broadside (phi = 0); used only
    as an independent check at intermediate angles.
    """
    wavelength = C0 / freq
    k = 2.0 * np.pi / wavelength
    phi = np.radians(np.atleast_1d(phi_deg))
    pre = -np.exp(-1j * np.pi / 4.0) / (2.0 * np.sqrt(2.0 * np.pi * k))
    # Angle from each edge's face to the radar direction
    phi_top = np.pi / 2.0 + phi      # edge at y = +width/2, face toward -y
    phi_bot = np.pi / 2.0 - phi      # edge at y = -width/2, face toward +y
    ph_top = np.exp(1j * k * width * np.sin(phi))
    ph_bot = np.exp(-1j * k * width * np.sin(phi))
    out = []
    for sign in (-1.0, +1.0):        # soft (VV), hard (HH)
        d_top = pre * (1.0 + sign / np.cos(phi_top))
        d_bot = pre * (1.0 + sign / np.cos(phi_bot))
        out.append(2.0 * np.pi * np.abs(d_top * ph_top + d_bot * ph_bot) ** 2)
    return tuple(out)


def to_3d(sigma_2d, lz, freq):
    """Convert 2D echo width to 3D RCS for a plate of length lz along z."""
    return sigma_2d * 2.0 * lz ** 2 / (C0 / freq)


def db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def validate(ly, lz, freq):
    """Print checks of the MoM edge-diffraction model."""
    wavelength = C0 / freq
    k = 2.0 * np.pi / wavelength
    print(f"Validation: strip width {ly * 100:.1f} cm ({ly / wavelength:.2f} lambda), "
          f"f = {freq / 1e9:.2f} GHz")

    vv, hh = strip_echo_width_mom(ly, freq, [0.0])
    po_2d = k * ly ** 2
    print("\n1) Broadside 2D echo width vs PO (k w^2):")
    print(f"   PO {db(po_2d):.3f} dBm,  MoM VV {db(vv[0]):.3f} dBm,  MoM HH {db(hh[0]):.3f} dBm")

    angles = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0, 90.0])
    vv1, hh1 = strip_echo_width_mom(ly, freq, angles, segments_per_wavelength=40)
    vv2, hh2 = strip_echo_width_mom(ly, freq, angles, segments_per_wavelength=80)
    print("\n2) Mesh convergence, 40 vs 80 segments per wavelength (dBm):")
    print("   phi    VV(40)   VV(80)    HH(40)   HH(80)")
    for row in zip(angles, db(vv1), db(vv2), db(hh1), db(hh2)):
        print("   {:4.0f}  {:7.2f}  {:7.2f}   {:7.2f}  {:7.2f}".format(*row))

    peaks = np.arange(20.0, 81.0, 0.25)
    vv_m, hh_m = strip_echo_width_mom(ly, freq, peaks)
    vv_g, hh_g = strip_echo_width_gtd(ly, freq, peaks)
    print("\n3) MoM vs single-diffraction GTD, 20-80 deg (dBm):")
    print("   phi    VV MoM   VV GTD    HH MoM   HH GTD")
    for ang in (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0):
        i = np.argmin(np.abs(peaks - ang))
        print("   {:4.0f}  {:7.2f}  {:7.2f}   {:7.2f}  {:7.2f}".format(
            peaks[i], db(vv_m[i]), db(vv_g[i]), db(hh_m[i]), db(hh_g[i])))
    # Compare lobe envelopes (away from nulls) by RMS difference
    for name, m, g in (("VV", vv_m, vv_g), ("HH", hh_m, hh_g)):
        ok = (m > m.max() * 1e-2) & (g > g.max() * 1e-2)
        print(f"   {name}: RMS |MoM - GTD| over lobes = {np.sqrt(np.mean((db(m[ok]) - db(g[ok])) ** 2)):.2f} dB")

    vv3, hh3 = strip_echo_width_mom(ly, freq, [0.0])
    print("\n4) 3D broadside RCS:")
    print(f"   PO 4 pi A^2 / lambda^2 = {db(po_closed_form_rcs(ly, lz, freq, [0.0])[0]):.2f} dBsm,  "
          f"MoM VV {db(to_3d(vv3[0], lz, freq)):.2f} dBsm,  MoM HH {db(to_3d(hh3[0], lz, freq)):.2f} dBsm")


def main():
    parser = argparse.ArgumentParser(description="Monostatic RCS of a PEC plate in the y-z plane vs azimuth.")
    parser.add_argument("--ly", type=float, default=0.12, help="plate length along y in meters (default 0.12)")
    parser.add_argument("--lz", type=float, default=0.12, help="plate length along z in meters (default 0.12)")
    parser.add_argument("--freq", type=float, default=10e9, help="frequency in Hz (default 10e9)")
    parser.add_argument("--points", type=int, default=721, help="number of azimuth samples (default 721)")
    parser.add_argument("--curves", type=str, default="po,vv,hh",
                        help="comma-separated curves to plot: po (numerical PO), po-cf (closed-form PO), "
                             "vv (MoM, E along z), hh (MoM, E in x-y plane). Default po,vv,hh")
    parser.add_argument("--save", type=str, default=None, help="save plot to this file instead of showing it")
    parser.add_argument("--validate", action="store_true", help="run the MoM validation checks and exit")
    args = parser.parse_args()

    if args.validate:
        validate(args.ly, args.lz, args.freq)
        return

    curves = [c.strip().lower() for c in args.curves.split(",") if c.strip()]
    unknown = set(curves) - {"po", "po-cf", "vv", "hh"}
    if unknown:
        parser.error(f"unknown curve(s): {', '.join(sorted(unknown))}")

    phi_deg = np.linspace(-90.0, 90.0, args.points)
    wavelength = C0 / args.freq
    results = {}
    if "po" in curves:
        results["po"] = po_monostatic_rcs(args.ly, args.lz, args.freq, phi_deg)
    if "po-cf" in curves:
        results["po-cf"] = po_closed_form_rcs(args.ly, args.lz, args.freq, phi_deg)
    if "vv" in curves or "hh" in curves:
        vv, hh = strip_echo_width_mom(args.ly, args.freq, phi_deg)
        results["vv"] = to_3d(vv, args.lz, args.freq)
        results["hh"] = to_3d(hh, args.lz, args.freq)

    peak = 4.0 * np.pi * (args.ly * args.lz) ** 2 / wavelength ** 2
    print(f"Plate {args.ly * 100:.1f} cm (y) x {args.lz * 100:.1f} cm (z) in the y-z plane")
    print(f"Frequency f = {args.freq / 1e9:.3f} GHz (lambda = {wavelength * 100:.3f} cm), "
          f"L_y / lambda = {args.ly / wavelength:.2f}")
    print(f"Normal-incidence PO RCS 4 pi A^2 / lambda^2 = {peak:.4f} m^2 ({db(peak):.2f} dBsm)")
    shown = [c for c in ("po", "po-cf", "vv", "hh") if c in curves]
    print("   phi  " + "".join(f"{c.upper():>10s}" for c in shown) + "   (dBsm)")
    for ang in (0.0, 10.0, 30.0, 60.0, 80.0, 90.0):
        i = np.argmin(np.abs(phi_deg - ang))
        print(f"  {phi_deg[i]:5.1f}" + "".join(f"{db(results[c][i]):10.2f}" for c in shown))

    styles = {
        "po": dict(color="gray", ls="--", lw=1.5, label="Physical optics"),
        "po-cf": dict(color="black", ls=":", lw=1.5, label="Physical optics (closed form)"),
        "vv": dict(color="tab:blue", lw=2, label="MoM edge diffraction, VV (E ∥ z)"),
        "hh": dict(color="tab:orange", lw=2, label="MoM edge diffraction, HH (E in x-y)"),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in shown:
        ax.plot(phi_deg, db(results[c]), **styles[c])
    ax.set_xlim(-90, 90)
    ax.set_xticks(np.arange(-90, 91, 30))
    top = db(peak)
    ax.set_ylim(top - 60, top + 5)
    ax.set_xlabel("Azimuth angle of incidence, $\\phi$ (deg)")
    ax.set_ylabel("Monostatic RCS (dBsm)")
    ax.set_title(
        f"PEC plate monostatic RCS: {args.ly * 100:.1f} cm x {args.lz * 100:.1f} cm "
        f"(y-z plane), f = {args.freq / 1e9:.2f} GHz"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Plot saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
