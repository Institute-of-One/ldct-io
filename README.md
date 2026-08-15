# ldct-io

**Read, reconstruct and measure real CT projection data.** The bridge between a public
projection archive and the task-based image-quality framework in
[taskiq-core](https://github.com/Institute-of-One/taskiq-core) and
[denoiq-core](https://github.com/Institute-of-One/denoiq-core).

Those two packages are deliberately free of DICOM and of patient data, and that is a property
worth keeping: it is why every number in them can be checked against a closed form. This
package is where the archive, its private tags and its traps live, so neither of them has to
change to meet real data.

> **Status: v0.1.0.** Geometry, series reading, single-slice rebinning, equiangular fan-beam
> FBP and MTF from a circular edge are implemented and validated against closed forms.
> Rebinning that is correct for an object varying along z (Noo et al.) is not yet written.

---

## The one idea, carried over

**Every estimator is validated against a closed-form answer, not against itself.**

A uniform disk has an analytic fan-beam projection, so its reconstruction has a known answer
everywhere. The test suite reconstructs one through the same code path the real data takes:

| quantity | truth | measured |
|---|---|---|
| centre of the disk | 0 HU | **+0.02 HU** |
| outside the disk | −1000 HU | **−1000.0 HU** |
| fitted radius | 99.939 mm | 99.890 mm (**−49 µm**) |
| ellipticity of a disk | 0 | < 20 µm |
| MTF50 | the sampling chain | within 30 % |

That is not decoration. A disk is also the exact model of a *cylindrical phantom in a helical
scan*, because the object does not vary along z: every ray sees the same object regardless of
its axial position, so single-slice rebinning of a helical acquisition and a circular scan of
the same object produce the same sinogram. A circular-scan closed form therefore validates a
helical pipeline.

## Two traps this package exists to prevent

**1. File-name order is not acquisition order.** A series downloaded from an archive arrives
with sequential file names (`00000001.dcm`, …) assigned in an order unrelated to the scan:

```
00009001.dcm  ->  InstanceNumber  2623
00009002.dcm  ->  InstanceNumber  6002
00000001.dcm  ->  InstanceNumber  8373
```

Reading in file-name order interleaves unrelated gantry angles. The insidious part is that the
per-view geometry tags, read the same way and then sorted, still look perfect — a uniform
angular step with a standard deviation of 8 × 10⁻⁸ rad — so nothing warns you. The error
surfaces only as a sinogram that will not reconstruct. `index_series` orders by
`InstanceNumber` and will not hand out frames any other way; `tests/test_reading.py` fails if
that regresses.

**2. Dose tags on simulated low-dose series are inherited, not recomputed.** In
LDCT-and-Projection-data the "Low Dose Images" of a case carry a *higher* `XRayTubeCurrent`
and `Exposure` than the full-dose series they were simulated from. Reading dose from the
header gives the wrong answer with no indication that anything is wrong.

## Also load-bearing

- **A geometry it has not been validated against is refused.** A flat detector, an in-plane
  flying focal spot, a source-to-detector distance shorter than the source-to-isocentre
  distance — each raises rather than producing a plausible image.
- **The flying focal spot is handled exactly, not approximately.** A z-flying spot cycles
  through a fixed set of positions, so the views split into classes each of which has a
  constant, exactly known geometry. `ViewTable.focal_spot_classes()` returns that split.
  Reconstructing one class is exact; the alternative — a projector with a per-view source
  position — is what most toolkits cannot express.
- **A short scan is refused.** Under-covered views reconstruct to a plausible image with the
  wrong contrast. Parker weighting is not implemented, so the function raises instead.
- **A missing view is refused.** A gap in the series is invisible to a reconstruction; it only
  streaks.
- **An edge whose background is not flat is refused.** A sloped pedestal survives
  differentiation and, through the DC normalisation, drags the whole MTF down. Real CT images
  have such slopes. `background="asymptote"` removes them deliberately, and says so in `meta`.
- **CPU only.** One slice reconstructs in about 20 s on a laptop. A reconstruction a reviewer
  cannot run without a CUDA build is not reproducible.

## Install

```bash
pip install -e .
```

Python ≥ 3.10. Runtime dependencies are numpy, scipy and pydicom.

## Use

```python
from ldct_io import index_series, single_slice_rebin, fan_beam_fbp, to_hu

series = index_series(r"D:\DevData\TCIA\LDCT-and-Projection-data\ACR_Phantom\projections")
print(len(series), series.views.views_per_rotation, series.views.pitch(series.geometry))
# 18032  2304.1  0.7985

sino = single_slice_rebin(series, z=-170.0, focal_spot_class=0)
print(sino.drift_at_radius(100.0))          # how far the rays wander in z out there

recon = fan_beam_fbp(sino.sinogram, series.geometry, sino.angles, fov=260.0, n_pixels=512)
hu = to_hu(recon.image, series.geometry.water_attenuation)
```

```python
from ldct_io import fit_edge_circle, radial_mtf

circle = fit_edge_circle(hu, recon.spacing, search_range=(85.0, 112.0))
mtf = radial_mtf(hu, recon.spacing, circle, band=6.0, background="asymptote")
print(circle.radius, mtf.mtf50, mtf.meta["tail_slope_inner_fraction"])
```

NPS and NEQ are not reimplemented here — pass the reconstructed ROIs straight to
`taskiq_core.nps_2d` and `taskiq_core.neq`.

## A caution about MTF and NPS in CT

They do not share a transfer chain. Quantum noise is generated *at the detector*, after the
object and after the focal spot, so the transfer implied by the noise excludes the focal-spot
blur that the signal has already been through. On real data the two differ by a factor of two
at 100 mm from the isocentre, and that is physics, not a bug — but it does mean an NPS cannot
be used to validate an MTF. Only a closed form can.

## API

| Module | Contents |
|---|---|
| `ldct_io.geometry` | `ScanGeometry`, `ViewTable`, the DICOM-CT-PD private tags, focal-spot classes |
| `ldct_io.dicomctpd` | `index_series` (acquisition order), `ProjectionSeries`, `directory_digest` |
| `ldct_io.rebin` | `single_slice_rebin` — helical acquisition to the sinogram of one plane |
| `ldct_io.recon` | `fan_beam_fbp`, `filter_projections`, `backproject`, `ramp_kernel`, `to_hu` |
| `ldct_io.phantoms` | `disk_sinogram`, `sampling_chain_mtf` — the closed-form references |
| `ldct_io.edge` | `fit_edge_circle`, `radial_mtf` — MTF from a circular edge |
| `ldct_io.manifest` | `Manifest`, `SeriesRecord` — which series were used, under what licence |

## Data, and what is deliberately excluded

The data lives outside this repository; what travels in git is the manifest of
`SeriesInstanceUID`s that identifies it. `.gitignore` and the CI refuse `.dcm` outright.

This project reads only the **chest, liver/abdomen and phantom** parts of
[LDCT-and-Projection-data](https://doi.org/10.7937/9npb-2637), which are CC BY 4.0. The
**head cases are excluded by design**: they remain under the NIH Controlled Data Access Policy
because a face can be reconstructed from them. `SeriesRecord` raises if a head series is put
into a manifest.

## Licence

MIT — see [`LICENSE`](LICENSE). The helical-to-fan-beam rebinning geometry follows
[helix2fan](https://github.com/faebstn96/helix2fan) (Apache-2.0), whose DICOM-CT-PD tag mapping
was the starting point for `ldct_io.geometry`.
