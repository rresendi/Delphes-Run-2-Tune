import argparse
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep
from collections import defaultdict
from scipy.stats import norm

plt.style.use(hep.style.CMS)


def ffloat(x):
    s = str(x).strip()
    if s in ("", "nan", "NaN"):
        return float("nan")
    return float(s)


def read_summary(path):
    bins = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            bins.append({
                "eta_lo": ffloat(row["eta_lo"]),
                "eta_hi": ffloat(row["eta_hi"]),
                "pt_lo":  ffloat(row["pt_lo"]),
                "pt_hi":  ffloat(row["pt_hi"]),
                "n":      int(float(row["n"])) if row["n"].strip() else 0,
                "mu":     ffloat(row["mu"]),
                "sigma":  ffloat(row["sigma"]),
            })
    return bins


def find_bin(bins, eta, pt):
    for b in bins:
        if b["eta_lo"] <= abs(eta) < b["eta_hi"] and b["pt_lo"] <= pt < b["pt_hi"]:
            return b
    return None


def bin_key(b):
    return (b["eta_lo"], b["eta_hi"], b["pt_lo"], b["pt_hi"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csv")
    ap.add_argument("summary_csv")
    ap.add_argument("--outdir",  default=".")
    ap.add_argument("--prefix",  default="resolution")
    ap.add_argument("--bins",    type=int, default=80, help="histogram bins (default 80)")
    ap.add_argument("--xlo",     type=float, default=-0.5)
    ap.add_argument("--xhi",     type=float, default=0.5)
    args = ap.parse_args()

    summary = read_summary(args.summary_csv)

    eta_bins = sorted({(b["eta_lo"], b["eta_hi"]) for b in summary})
    pt_bins  = sorted({(b["pt_lo"],  b["pt_hi"])  for b in summary})

    # collect raw resolution per bin
    raw = defaultdict(list)
    overflow  = defaultdict(int)
    underflow = defaultdict(int)

    with open(args.raw_csv, newline="") as f:
        for row in csv.DictReader(f):
            eta = ffloat(row["gen_eta"])
            pt  = ffloat(row["gen_pt"])
            res = ffloat(row["resolution"])
            if not (math.isfinite(eta) and math.isfinite(pt) and math.isfinite(res)):
                continue
            b = find_bin(summary, eta, pt)
            if b is None:
                continue
            key = bin_key(b)
            if res < args.xlo:
                underflow[key] += 1
            elif res > args.xhi:
                overflow[key] += 1
            else:
                raw[key].append(res)

    n_eta = len(eta_bins)
    n_pt  = len(pt_bins)

    fig, axes = plt.subplots(
        n_eta, n_pt,
        figsize=(4 * n_pt, 4 * n_eta),
        sharey=False,
    )
    # ensure 2-D indexing even for 1-row/1-col cases
    if n_eta == 1: axes = axes[np.newaxis, :]
    if n_pt  == 1: axes = axes[:, np.newaxis]

    bin_edges = np.linspace(args.xlo, args.xhi, args.bins + 1)
    bin_w     = bin_edges[1] - bin_edges[0]

    for i, (elo, ehi) in enumerate(eta_bins):
        for j, (plo, phi) in enumerate(pt_bins):
            ax  = axes[i][j]
            key = (elo, ehi, plo, phi)

            vals = np.array(raw[key])
            uf   = underflow[key]
            of   = overflow[key]
            n_total = len(vals) + uf + of

            # histogram (in-range values)
            counts, _ = np.histogram(vals, bins=bin_edges)

            # merge under/overflow into first/last bin
            counts_display = counts.copy()
            counts_display[0]  += uf
            counts_display[-1] += of

            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.bar(centers, counts_display, width=bin_w, color="steelblue",
                   alpha=0.7, label=f"N={n_total}")

            # mark overflow/underflow bins with hatching
            if uf > 0:
                ax.bar(centers[0], counts_display[0], width=bin_w,
                       color="none", edgecolor="red", hatch="//", linewidth=1.2)
            if of > 0:
                ax.bar(centers[-1], counts_display[-1], width=bin_w,
                       color="none", edgecolor="red", hatch="//", linewidth=1.2)

            # Gaussian overlay
            b_info = next((b for b in summary if bin_key(b) == key), None)
            if b_info and math.isfinite(b_info["mu"]) and math.isfinite(b_info["sigma"]) and b_info["sigma"] > 0:
                mu, sig = b_info["mu"], b_info["sigma"]
                x_fit   = np.linspace(args.xlo, args.xhi, 400)
                y_fit   = norm.pdf(x_fit, mu, sig) * n_total * bin_w
                ax.plot(x_fit, y_fit, "r-", linewidth=1.8,
                        label=rf"$\mu$={mu:.3f}, $\sigma$={sig:.3f}")
                # 3σ lines
                for nsig, ls in [(3, "--"), (4, ":")]:
                    for sign in [-1, 1]:
                        xv = mu + sign * nsig * sig
                        if args.xlo <= xv <= args.xhi:
                            ax.axvline(xv, color="gray", linestyle=ls, linewidth=0.9,
                                       label=f"±{nsig}σ" if sign == 1 else None)

            ax.set_title(
                rf"$|\eta|\in[{elo:g},{ehi:g})$,  $p_T\in[{plo:g},{phi:g})$ GeV",
                fontsize=9,
            )
            ax.set_xlabel(r"$(p_T^{\rm reco} - p_T^{\rm gen})\,/\,p_T^{\rm gen}$", fontsize=8)
            ax.set_ylabel("Entries", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(5))

            # annotate over/underflow counts
            if uf > 0:
                ax.text(0.02, 0.97, f"UF: {uf}", transform=ax.transAxes,
                        fontsize=7, va="top", color="red")
            if of > 0:
                ax.text(0.98, 0.97, f"OF: {of}", transform=ax.transAxes,
                        fontsize=7, va="top", ha="right", color="red")

    fig.suptitle(f"{args.prefix} — momentum resolution per bin", fontsize=13, y=1.01)
    fig.tight_layout()

    out = f"{args.outdir}/{args.prefix}_resolution_bins.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
