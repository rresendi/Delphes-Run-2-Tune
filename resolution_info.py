import argparse
import csv
import math
from collections import defaultdict

import numpy as np


ETA_EDGES = (0.0, 0.80, 1.44, 1.98, 2.50)


def make_bins_from_edges(edges):
    edges = list(edges)
    if len(edges) < 2:
        raise ValueError("Need at least 2 edges")
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


ETA_BINS = make_bins_from_edges(ETA_EDGES)


def find_bin(x, bins):
    for k, (lo, hi) in enumerate(bins):
        if lo <= x < hi:
            return k
    return None


def make_pt_bins_from_edges(edges):
    edges = list(edges)
    if len(edges) < 2:
        raise ValueError("Need at least 2 pT edges")
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def find_pt_bin(pt, pt_bins):
    for k, (plo, phi) in enumerate(pt_bins):
        if plo <= pt < phi:
            return k
    return None


def parse_edges(s):
    # accepts "1,2,3,4" or "1 2 3 4"
    toks = [t for t in s.replace(",", " ").split() if t]
    return [float(x) for x in toks]


def sample_pt_bins(sample):
    sample = sample.lower()
    if sample == "jpsi":
        edges = [1, 5, 6, 7, 8, 10]
    elif sample == "dy":
        edges = [10.0, 100.0, 1000.0]
    elif sample == "zprime":
        edges = [1000.0, 3000.0]
    else:
        raise ValueError("sample must be one of: jpsi, dy, zprime")
    return make_pt_bins_from_edges(edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", help="*_momentum_resolution.csv")
    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="output summary CSV (default: <input>_summary.csv)",
    )

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", choices=["jpsi", "dy", "zprime"], help="use standard pT bins")
    g.add_argument(
        "--pt-edges",
        type=str,
        help='custom pT edges, e.g. "1,2,3,4,5,6,7,8,10"',
    )

    ap.add_argument(
        "--pt-field",
        choices=["gen_pt", "reco_pt"],
        default="gen_pt",
        help="which pT to use for binning (default: gen_pt)",
    )

    ap.add_argument(
        "--eta-field",
        choices=["reco_eta", "gen_eta"],
        default="reco_eta",
        help="which eta to use for eta binning (default: reco_eta)",
    )

    ap.add_argument(
        "--eta-abs",
        action="store_true",
        default=True,
        help="bin in abs(eta) (default: True). Use --no-eta-abs to disable.",
    )
    ap.add_argument(
        "--no-eta-abs",
        dest="eta_abs",
        action="store_false",
        help="bin in signed eta instead of abs(eta).",
    )

    args = ap.parse_args()

    if args.sample:
        pt_bins = sample_pt_bins(args.sample)
    else:
        pt_bins = make_pt_bins_from_edges(parse_edges(args.pt_edges))

    out_path = args.output
    if out_path is None:
        out_path = args.input_csv[:-4] + "_summary.csv" if args.input_csv.lower().endswith(".csv") else args.input_csv + "_summary.csv"

    # Collect resolutions per (eta_lo, eta_hi, pt_bin_index)
    groups = defaultdict(list)

    with open(args.input_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        required = {args.eta_field, "resolution", args.pt_field}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise RuntimeError(f"Input missing columns: {sorted(missing)}")

        for row in r:
            try:
                eta = float(row[args.eta_field])
                if args.eta_abs:
                    eta = abs(eta)

                pt = float(row[args.pt_field])
                res = float(row["resolution"])
            except (ValueError, TypeError, KeyError):
                continue

            ie = find_bin(eta, ETA_BINS)
            if ie is None:
                continue  # outside [0,2.4) (or outside signed range if --no-eta-abs)

            ip = find_pt_bin(pt, pt_bins)
            if ip is None:
                continue

            eta_lo, eta_hi = ETA_BINS[ie]
            groups[(eta_lo, eta_hi, ip)].append(res)

    # Write summary (loop over ALL fixed eta bins and pt bins)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eta_lo", "eta_hi", "pt_lo", "pt_hi", "n", "mu", "mu_err", "sigma", "sigma_err"])

        for (eta_lo, eta_hi) in ETA_BINS:
            for ip, (pt_lo, pt_hi) in enumerate(pt_bins):
                vals = groups.get((eta_lo, eta_hi, ip), [])
                n = len(vals)

                if n == 0:
                    w.writerow([f"{eta_lo:.6f}", f"{eta_hi:.6f}", f"{pt_lo:.6f}", f"{pt_hi:.6f}", 0, "", "", "", ""])
                    continue

                arr = np.asarray(vals, dtype=float)
                mu = float(np.mean(arr))
                sigma = float(np.std(arr, ddof=1)) if n >= 2 else float("nan")

                mu_err = sigma / math.sqrt(n) if (n >= 2 and math.isfinite(sigma)) else float("nan")
                sigma_err = sigma / math.sqrt(2.0 * (n - 1)) if (n >= 3 and math.isfinite(sigma)) else float("nan")

                w.writerow(
                    [
                        f"{eta_lo:.6f}",
                        f"{eta_hi:.6f}",
                        f"{pt_lo:.6f}",
                        f"{pt_hi:.6f}",
                        n,
                        f"{mu:.6f}",
                        "" if not math.isfinite(mu_err) else f"{mu_err:.6f}",
                        "" if not math.isfinite(sigma) else f"{sigma:.6f}",
                        "" if not math.isfinite(sigma_err) else f"{sigma_err:.6f}",
                    ]
                )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
