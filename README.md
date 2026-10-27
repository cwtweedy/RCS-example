# RCS-example

Python scripts that compute the **monostatic radar cross section (RCS)** of
simple perfectly electrically conducting (PEC) targets and plot it against
the azimuth angle of the incident plane wave.

Every script uses the same setup:

- The target is centered at the origin.
- The radar direction rotates in the x-y plane, with azimuth φ swept from −90° to +90°.
  φ = 0 means looking along the +x axis.
- The radar receives in the backscatter direction (monostatic).
- The default frequency is 10 GHz (λ ≈ 3 cm).

| Script | Target (default) | Method | Example plot |
| --- | --- | --- | --- |
| [`sphere_rcs.py`](sphere_rcs.py) | Sphere, radius 3 cm | Exact Mie series | [`rcs_plot.png`](rcs_plot.png) |
| [`ellipsoid_rcs.py`](ellipsoid_rcs.py) | Prolate ellipsoid, 6 cm × 3 cm, major axis along y | Physical optics + geometric optics | [`ellipsoid_rcs_plot.png`](ellipsoid_rcs_plot.png) |
| [`plate_rcs.py`](plate_rcs.py) | Flat plate, 12 cm × 12 cm, in the y-z plane | Physical optics + 2D method-of-moments edge diffraction | [`plate_rcs_plot.png`](plate_rcs_plot.png) |

## Setup

The scripts need Python 3 (developed and tested on 3.13) and the packages
in [`requirements.txt`](requirements.txt): numpy, scipy and matplotlib.

```bash
python -m venv .venv
```

Activate the environment: `.venv\Scripts\activate` on Windows, or
`source .venv/bin/activate` on macOS and Linux. Then install the packages:

```bash
pip install -r requirements.txt
```

## Usage

Each script opens an interactive plot window by default. Pass `--save FILE.png` to
write the plot to a file instead. Each one also prints key values to the
terminal. All lengths are in meters and frequencies in hertz. Run any
script with `-h` to list its options.

Options shared by all three scripts:

| Option | Meaning | Default |
| --- | --- | --- |
| `--freq` | Frequency in Hz | `10e9` |
| `--points` | Number of azimuth samples from −90° to +90° | 361 (721 for the plate) |
| `--save` | Save the plot to this file instead of showing it | (show window) |

### Sphere: `sphere_rcs.py`

```bash
python sphere_rcs.py --radius 0.03
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--radius` | Sphere radius in m | `0.03` |

This script uses the exact Mie series solution. Because a sphere looks the same from every
direction, the monostatic RCS is the same at every azimuth, so the plot is
a flat line. For a 3 cm sphere at 10 GHz (ka ≈ 6.3) the result is
**−25.45 dBsm**, about 1.008 × πa². The plot also shows the optical limit πa².

### Ellipsoid: `ellipsoid_rcs.py`

```bash
python ellipsoid_rcs.py --major 0.06 --minor 0.03
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--major` | Full length of the major axis (along y) in m | `0.06` |
| `--minor` | Full length of the two minor axes (along x and z) in m | `0.03` |
| `--validate` | Compare physical optics with the Mie series for a 3 cm sphere, then exit | off |

There is no simple exact solution for an ellipsoid, so the script plots
two high-frequency approximations:

- **Physical optics (PO):** numerically integrates the induced surface current over the lit half
  of the surface.
- **Geometric optics (GO):** the closed-form specular-point result,
  σ = π a_x² a_y² a_z² / (a_x² cos²φ + a_y² sin²φ)².

With the default size, the RCS falls from about −25.5 dBsm broadside
(φ = 0) to −37.5 dBsm end-on (φ = ±90°). Both methods ignore creeping
waves, so expect errors of up to about 1 dB at this electrical size.
For a sphere, PO agrees with the exact Mie result to within 0.1 dB.

### Plate: `plate_rcs.py`

```bash
python plate_rcs.py --ly 0.12 --lz 0.12
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--ly` | Plate length along y in m | `0.12` |
| `--lz` | Plate length along z in m | `0.12` |
| `--curves` | Comma-separated curves to plot: `po`, `po-cf`, `vv`, `hh` | `po,vv,hh` |
| `--validate` | Run the edge-diffraction checks and exit | off |

The plate is infinitely thin and lies in the y-z plane, so φ = 0 is
normal incidence and φ = ±90° is edge-on. The script offers these curves:

- **`po`: physical optics, numerically integrated.**
- **`po-cf`: closed-form physical optics.** σ = (4πA²/λ²) cos²φ [sin X / X]², where X = k L_y sin φ.
  Physical optics ignores edge diffraction, so it gives the same answer
  for both polarizations and drops to zero at grazing angles.
- **`vv`, `hh`: edge-diffraction model.** The plate's cross-section is treated as a strip of width L_y. Its 2D echo width
  is solved with a method-of-moments solution of the electric-field
  integral equation, which captures all diffraction from the two edges
  parallel to z. The result is scaled to 3D with σ₃D = σ₂D · 2 L_z² / λ.
  - **VV:** E parallel to z, along the diffracting edges.
  - **HH:** E in the x-y plane, across the diffracting edges.

At normal incidence all models give about **4.6 dBsm** (4πA²/λ²). At wide
angles the polarizations separate. VV levels off near −23 dBsm. HH shows
a wide-angle lobe near ±65°, caused by diffraction bouncing between the edges.
The 2D-to-3D scaling ignores diffraction from the edges parallel to y and
from the corners, which is weak in this plane.

## Validation

Two of the scripts include self-checks, run with `--validate`:

```bash
python ellipsoid_rcs.py --validate
```

```bash
python plate_rcs.py --validate
```

- **Sphere:** no `--validate` option. During development, the Mie result was checked
  against the low-frequency (Rayleigh) limit 9πa²(ka)⁴ and the high-frequency
  (optical) limit πa², and matched both.
- **Ellipsoid:** PO is compared against the exact Mie series for a sphere
  and agrees to within 0.1 dB.
- **Plate:** the method-of-moments solution is checked three ways:
  - Against physical optics at normal incidence: within 0.1 dB.
  - Against a mesh twice as fine: within about 0.2 dB.
  - Against single-diffraction GTD (geometrical theory of diffraction) at 20° to 80°: VV agrees to 0.09 dB RMS.
    HH differs by about 3 dB, the expected effect of diffraction bouncing
    between the edges, which single-diffraction GTD leaves out.
