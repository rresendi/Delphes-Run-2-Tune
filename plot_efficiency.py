#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

csv_file = "/eos/user/r/rresendi/results/zp/zprime_muon_efficiency.csv"

# Group rows by eta bin
groups = {}  # key: (eta_lo, eta_hi) -> list of dict rows
with open(csv_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        eta_lo = float(row["eta_lo"])
        eta_hi = float(row["eta_hi"])
        key = (eta_lo, eta_hi)
        groups.setdefault(key, []).append(row)

# CMS style
hep.style.use("CMS")

for (eta_lo, eta_hi), rows in sorted(groups.items()):
    # Sort by pt_lo just in case
    rows = sorted(rows, key=lambda r: float(r["pt_lo"]))

    pt_lo = np.array([float(r["pt_lo"]) for r in rows], dtype=float)
    pt_hi = np.array([float(r["pt_hi"]) for r in rows], dtype=float)
    eff   = np.array([float(r["eff"])   for r in rows], dtype=float)
    err_lo = np.array([float(r["err_lo"]) for r in rows], dtype=float)
    err_up = np.array([float(r["err_up"]) for r in rows], dtype=float)

    # Bin centers and widths
    pt_centers = 0.5 * (pt_lo + pt_hi)
    pt_err = 0.5 * (pt_hi - pt_lo)
    yerr = np.vstack([err_lo, err_up])

    fig, ax = plt.subplots(figsize=(8, 8))
    hep.cms.label(ax=ax, data=True, label="Preliminary")

    ax.errorbar(
        pt_centers,
        eff,
        xerr=pt_err,
        yerr=yerr,
        fmt="o",
        capsize=2,
        linewidth=1.5,
        markersize=5,
        color="green",
        ecolor="green",
    )

    # Eta label on canvas
    ax.text(
        0.08, 0.08,
        rf"${eta_lo:.1f} < |\eta| < {eta_hi:.1f}$",
        transform=ax.transAxes,
        fontsize=17,
        verticalalignment="bottom",
    )

    ax.set_xlabel(r"$p_T^{\mu}$ [GeV]")
    ax.set_ylabel("Efficiency")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(pt_lo.min() - 0.2, pt_hi.max() + 0.2)

    # If you prefer log-x for the 10→1000 bins, uncomment:
    # ax.set_xscale("log")

    fig.tight_layout()

    out_name = f"eff_vs_pt_eta_{eta_lo:.1f}_{eta_hi:.1f}.pdf"
    fig.savefig(out_name)
    plt.close(fig)
    print(f"Saved plot → {out_name}")
