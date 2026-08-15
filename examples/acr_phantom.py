"""Reconstruct the ACR phantom from real projections and measure the system MTF.

Needs the ACR_Phantom series of LDCT-and-Projection-data (CC BY 4.0, DOI 10.7937/9npb-2637)
on disk. Point ``--data`` at the directory of ``.dcm`` files.

    python examples/acr_phantom.py --data D:/DevData/.../ACR_Phantom/projections --z -170
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ldct_io import (
    Manifest,
    SeriesRecord,
    directory_digest,
    fan_beam_fbp,
    fit_edge_circle,
    index_series,
    radial_mtf,
    sampling_chain_mtf,
    single_slice_rebin,
    to_hu,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--z", type=float, default=-170.0, help="slice position [mm]")
    parser.add_argument("--fov", type=float, default=260.0)
    parser.add_argument("--pixels", type=int, default=512)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--png", type=Path, default=None)
    args = parser.parse_args()

    series = index_series(args.data)
    g = series.geometry
    v = series.views
    print(f"{len(series)} views, {v.views_per_rotation:.1f} per rotation, "
          f"{v.rotations:.3f} rotations, pitch {v.pitch(g):.4f}")
    print(f"geometry: SID {g.source_to_isocentre} mm, SDD {g.source_to_detector:.1f} mm, "
          f"{g.n_channels}x{g.n_rows}, {g.detector_shape}, {g.focal_spot_mode}")
    classes = v.focal_spot_classes()
    print(f"focal-spot classes: {[c.size for c in classes]}")
    for k, idx in enumerate(classes):
        print(f"  class {k}: dz {v.ffs_dz[idx][0]:+.3f} mm, drho {v.ffs_drho[idx][0]:+.3f} mm")

    sino = single_slice_rebin(series, args.z, focal_spot_class=0)
    print(f"\nrebinned at z = {sino.z} mm: {sino.meta['n_views']} views, "
          f"rows {sino.meta['detector_rows_used'][0]:.1f}..{sino.meta['detector_rows_used'][1]:.1f}")
    print(f"  ray drift in z at r = 100 mm: {sino.drift_at_radius(100.0):.2f} mm")

    recon = fan_beam_fbp(
        sino.sinogram, g, sino.angles, fov=args.fov, n_pixels=args.pixels
    )
    hu = to_hu(recon.image, g.water_attenuation)
    X, Y = recon.pixel_coordinates()
    r = np.hypot(X, Y)
    print(f"\ncentre 40 mm ROI : {hu[r < 20].mean():+.1f} HU (sd {hu[r < 20].std():.1f})")
    print(f"outside          : {hu[(r > 115) & (r < 130)].mean():+.1f} HU  (air = -1000)")

    circle = fit_edge_circle(hu, recon.spacing, search_range=(85.0, 112.0))
    print(f"\nedge circle: R = {circle.radius:.3f} mm (diameter {2 * circle.radius:.2f} mm), "
          f"residual sd {circle.residual_sd * 1000:.0f} um, ellipticity "
          f"{circle.ellipticity * 1000:.0f} um")

    mtf = radial_mtf(hu, recon.spacing, circle, band=6.0, background="asymptote")
    f = np.linspace(1e-6, 1.0, 500)
    chain = sampling_chain_mtf(g, f, recon.spacing)
    chain50 = float(np.interp(0.5, chain[::-1], f[::-1]))
    print(f"MTF50 measured   : {mtf.mtf50:.3f} cyc/mm")
    print(f"MTF10 measured   : {mtf.mtf10:.3f} cyc/mm")
    print(f"MTF50 the sampling chain alone allows : {chain50:.3f} cyc/mm")
    print(f"  background tails: {mtf.meta['tail_slope_inner_fraction'] * 100:+.2f} % inner, "
          f"{mtf.meta['tail_slope_outer_fraction'] * 100:+.2f} % outer of contrast")
    print(
        "  anything blurrier than the sampling chain is the scanner (focal spot, detector "
        "response), not the code: the pipeline reproduces a closed form to 0.02 HU."
    )

    if args.manifest:
        digest = directory_digest(args.data)
        m = Manifest(notes=f"ACR phantom, slice z = {args.z} mm")
        m.add(
            SeriesRecord(
                series_instance_uid=g.meta.get("series_instance_uid", ""),
                patient_id="PatientID",
                body_part="PHANTOM",
                description="ACR CT accreditation phantom, full-dose projections",
                manufacturer=g.meta.get("manufacturer", ""),
                role="system MTF and NPS",
                **digest,
            )
        )
        print(f"\nwrote {m.write(args.manifest)}")

    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        ext = [-args.fov / 2, args.fov / 2, -args.fov / 2, args.fov / 2]
        axes[0].imshow(hu, cmap="gray", vmin=-200, vmax=200, extent=ext)
        axes[0].set_title(f"z = {args.z:.0f} mm, W400 L0")
        axes[0].set_xlabel("x [mm]")
        axes[1].plot(mtf.frequency, mtf.mtf, lw=1.8, label=f"measured ({mtf.mtf50:.3f})")
        axes[1].plot(f, chain, "k--", lw=1.2, label=f"sampling chain ({chain50:.3f})")
        axes[1].axhline(0.5, color="0.85", lw=0.8)
        axes[1].set_xlim(0, 1.0)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_xlabel("spatial frequency [cycles/mm]")
        axes[1].set_ylabel("MTF")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.png, dpi=130)
        print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
