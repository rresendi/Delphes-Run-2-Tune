#!/usr/bin/env python3
import csv, numpy as np, matplotlib.pyplot as plt, mplhep as hep

csv_file = "/eos/user/r/rresendi/results/jpsi/jpsi_muon_m3_etaDepPt_endcap_momentum_resolution.csv"
delphes_csv = "/eos/user/r/rresendi/madanalysis5/JPSI_MU/Build/momentum_resolution.csv"
samp = "jpsi"  # "jpsi", "dy", or "zprime"

# ---- bins (not in csv) ----
if samp == "jpsi":
    eta_edges = [0.0, 1.2, 2.4]
    pt_bins_per_eta = [[(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,10)]] * 2
elif samp == "dy":
    eta_edges, pt_edges = [0.0, 0.9, 1.2, 2.1, 2.4], [10.0, 100.0, 1000.0]
    pt_bins_per_eta = [[(pt_edges[i], pt_edges[i+1]) for i in range(len(pt_edges)-1)] for _ in range(len(eta_edges)-1)]
else:  # zprime
    eta_edges, pt_edges = [0.0, 0.9, 1.2, 2.1, 2.4], [1000.0, 3000.0]
    pt_bins_per_eta = [[(pt_edges[i], pt_edges[i+1]) for i in range(len(pt_edges)-1)] for _ in range(len(eta_edges)-1)]

# ---- read + bin (two sources) ----
acc = {(i, p0, p1): {"base": [], "delphes": []}
       for i in range(len(eta_edges)-1) for (p0, p1) in pt_bins_per_eta[i]}

def fill(path, key):
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            pt, res = float(r["gen_pt"]), float(r["resolution"])
            a = abs(float(r.get("gen_eta", r.get("eta", r.get("abs_gen_eta", "nan")))))
            if not np.isfinite(a): raise KeyError("Need gen_eta/eta/eta_gen column in the CSV.")
            i = next((k for k in range(len(eta_edges)-1) if eta_edges[k] <= a < eta_edges[k+1]), None)
            if i is None: continue
            j = next(((p0, p1) for (p0, p1) in pt_bins_per_eta[i] if p0 <= pt < p1), None)
            if j is None: continue
            acc[(i, *j)][key].append(res)

fill(csv_file, "base")
fill(delphes_csv, "delphes")

hep.style.use("CMS")

# ---- plot: counts vs resolution, with μ and σ in label ----
for (i, p0, p1), d in sorted(acc.items()):
    v0, v1 = np.asarray(d["base"]), np.asarray(d["delphes"])
    if v0.size == 0 and v1.size == 0: 
        continue

    n0, mu0, s0 = (v0.size, v0.mean(), v0.std()) if v0.size else (0, np.nan, np.nan)
    n1, mu1, s1 = (v1.size, v1.mean(), v1.std()) if v1.size else (0, np.nan, np.nan)

    fig, ax = plt.subplots(figsize=(8, 8))
    hep.cms.label(ax=ax, data=True, label="Preliminary")

    if v0.size:
        ax.hist(
            v0, bins=50, range=(-0.02, 0.02),
            histtype="stepfilled",
            label=rf"FullSim  $N={n0},\ \mu={mu0:.3g},\ \sigma={s0:.3g}$"
        )

    if v1.size:
        ax.hist(
            v1, bins=50, range=(-0.02, 0.02),
            histtype="stepfilled",
            label=rf"Delphes  $N={n1},\ \mu={mu1:.3g},\ \sigma={s1:.3g}$"
        )

    ax.set_xlim(-0.2, 0.2)
    ax.set(xlabel=r"$(p_T^\mathrm{reco}-p_T^\mathrm{gen})/p_T^\mathrm{gen}$", ylabel="Counts")
    ax.legend(frameon=False, fontsize=11)

    ax.text(0.05, 0.95,
            rf"{eta_edges[i]:.1f} <= |eta| < {eta_edges[i+1]:.1f}" "\n"
            rf"{p0:g} < $p_T^\mu$ < {p1:g} GeV",
            transform=ax.transAxes, va="top", fontsize=12)

    fig.tight_layout()
    out = f"res_hist_{samp}_overlay_eta_{eta_edges[i]:.1f}_{eta_edges[i+1]:.1f}_pt_{p0:g}_{p1:g}.pdf"
    fig.savefig(out); plt.close(fig); print("Saved →", out)
