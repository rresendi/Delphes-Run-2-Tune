#!/usr/bin/env python3
import ROOT
import os
import sys
import csv
import numpy as np

ROOT.gROOT.SetBatch(True)

# Calculate Clopper Pearson Uncertainty
CL = 0.683
ALPHA = 1.0 - CL


def clopper_pearson(num, den):
    eff = num / den if den > 0 else float("nan")
    if den <= 0:
        return eff, 0.0, 0.0
    low = ROOT.TEfficiency.ClopperPearson(int(den), int(num), ALPHA, False)
    up = ROOT.TEfficiency.ClopperPearson(int(den), int(num), ALPHA, True)
    return eff, eff - low, up - eff


# Helper Functions

# Pick best OSSF gen pair in mass window 
def pick_best_gen_pair(gen_objs, samp, resonance_mass):
    if samp == "jpsi":
        mwin = (2.9, 3.3)
    elif samp == "dy":
        mwin = (60.0, 120.0)
    else:  # zprime
        mwin = (4000.0, 6000.0)

    best_pair, best_delta = None, float("inf")
    for i, a in enumerate(gen_objs):
        for b in gen_objs[i + 1:]:
            # opposite sign, same flavor (muons)
            if a["pdgId"] * b["pdgId"] >= 0:
                continue
            p41 = ROOT.TLorentzVector()
            p41.SetPtEtaPhiM(a["pt"], a["eta"], a["phi"], a["mass"])
            p42 = ROOT.TLorentzVector()
            p42.SetPtEtaPhiM(b["pt"], b["eta"], b["phi"], b["mass"])
            m = (p41 + p42).M()
            if not (mwin[0] <= m <= mwin[1]):
                continue
            d = abs(m - resonance_mass)
            if d < best_delta:
                best_delta = d
                best_pair = (a, b)
    return best_pair

# Handle .txt or ROOT files
def read_file_list_or_root(path):
    out = []
    if os.path.isfile(path) and path.endswith(".txt"):
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                out.append(s)
    elif os.path.isfile(path) and path.endswith(".root"):
        out.append(path)
    elif os.path.isdir(path):
        for fn in os.listdir(path):
            if fn.endswith(".root"):
                out.append(os.path.join(path, fn))
    else:
        pass
    return out


# Main
def main():

    # System arguments/setup
    samp = sys.argv[1]
    if samp not in {"jpsi", "dy", "zprime"}:
        print("sample must be one of: jpsi, dy, zprime")
        sys.exit(1)

    require_medium = False
    inputs = []
    for a in sys.argv[2:]:
        if a == "--id":
            require_medium = True
        else:
            inputs.append(a)

    # Gather input files
    files = []
    for p in inputs:
        files.extend(read_file_list_or_root(p))
    if not files:
        print("No input ROOT files found.")
        sys.exit(1)

    # Sample-dependent pT/eta binning
    if samp == "jpsi":
        eta_edges = [0.0, 1.2, 2.4]
        jpsi_pt = [(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,10)]
        pt_bins_per_eta = [jpsi_pt, jpsi_pt]
        resonance_mass = 3.0969
    elif samp == "dy":
        eta_edges = [0.0, 0.9, 1.2, 2.1, 2.4]
        uniform_pt_edges = [10.0, 100.0, 1000.0]
        pt_bins_per_eta = [
            [
                (uniform_pt_edges[i], uniform_pt_edges[i + 1])
                for i in range(len(uniform_pt_edges) - 1)
            ]
            for _ in range(len(eta_edges) - 1)
        ]
        resonance_mass = 91.1876
    else:  # zprime
        eta_edges = [0.0, 0.9, 1.2, 2.1, 2.4]
        uniform_pt_edges = [1000.0, 3000.0]
        pt_bins_per_eta = [
            [
                (uniform_pt_edges[i], uniform_pt_edges[i + 1])
                for i in range(len(uniform_pt_edges) - 1)
            ]
            for _ in range(len(eta_edges) - 1)
        ]
        resonance_mass = 4800.0

    n_eta = len(eta_edges) - 1

    # Setup 2D lists for each eta/pT bin
    gen_totals = [[0] * len(pt_bins_per_eta[ie]) for ie in range(n_eta)]
    gen_matched = [[0] * len(pt_bins_per_eta[ie]) for ie in range(n_eta)]

    # Setup eta-binned momentum resolution lists
    momentum_resolutions = [[] for _ in range(n_eta)]  
    event_rows = [[] for _ in range(n_eta)]            

    # Event loop
    # Open and read input files
    for iz, fpath in enumerate(files):
        print(f"[{iz + 1}/{len(files)}] {fpath}")
        tf = ROOT.TFile.Open(fpath, "READ")
        if not tf or tf.IsZombie():
            print("Could not open. Skipping.")
            if tf:
                tf.Close()
            continue
        events = tf.Get("Events")
        if not events:
            print("No 'Events' tree. Skipping.")
            tf.Close()
            continue

        # Loop through events
        for ev in events:
            # Build list of gen muons
            nGen = int(getattr(ev, "nGenPart", 0))
            gp_pdg = getattr(ev, "GenPart_pdgId", None)
            gp_pt = getattr(ev, "GenPart_pt", None)
            gp_eta = getattr(ev, "GenPart_eta", None)
            gp_phi = getattr(ev, "GenPart_phi", None)
            gp_m = getattr(ev, "GenPart_mass", None)
            gp_st = getattr(ev, "GenPart_status", None)

            if any(x is None for x in (gp_pdg, gp_pt, gp_eta, gp_phi, gp_m, gp_st)):
                continue

            gen_objs = []
            # keep only stable muons within acceptance and overall pT envelope of this sample
            for i in range(nGen):
                if abs(int(gp_pdg[i])) != 13:
                    continue
                if int(gp_st[i]) != 1:
                    continue
                eta = float(gp_eta[i])
                if abs(eta) >= 2.4:
                    continue
                pt = float(gp_pt[i])
                if samp == "jpsi":
                    if not (1.0 <= pt < 10.0):
                        continue
                elif samp == "dy":
                    if not (10.0 <= pt < 1000.0):
                        continue
                else:
                    if not (1000.0 <= pt < 3000.0):
                        continue
                gen_objs.append(
                    {
                        "index": i,
                        "pt": pt,
                        "eta": eta,
                        "phi": float(gp_phi[i]),
                        "mass": float(gp_m[i]),
                        "pdgId": int(gp_pdg[i]),
                    }
                )

            if len(gen_objs) < 2:
                continue

            gen_objs.sort(key=lambda x: x["pt"], reverse=True)
            best = pick_best_gen_pair(gen_objs, samp, resonance_mass)
            if not best:
                continue
            pair = [best[0], best[1]]

            # Build reco muon collection
            nMu = int(getattr(ev, "nMuon", 0))
            mu_pt = getattr(ev, "Muon_pt", None)
            mu_eta = getattr(ev, "Muon_eta", None)
            mu_phi = getattr(ev, "Muon_phi", None)
            mu_m = getattr(ev, "Muon_mass", None)
            mu_med = getattr(ev, "Muon_mediumId", None)
            mu_gidx = getattr(ev, "Muon_genPartIdx", None)

            if any(x is None for x in (mu_pt, mu_eta, mu_phi, mu_m, mu_gidx)):
                continue

            # Do GenIndex matching
            reco_by_gen = {}
            for i in range(nMu):
                gidx = int(mu_gidx[i]) if i < len(mu_gidx) else -1
                if gidx < 0:
                    continue
                if require_medium and not bool(mu_med[i] if mu_med is not None else False):
                    continue
                # choose highest-pt candidate if multiple reco map to same gen
                prev = reco_by_gen.get(gidx)
                if (prev is None) or (mu_pt[i] > prev["pt"]):
                    reco_by_gen[gidx] = {
                        "pt": float(mu_pt[i]),
                        "eta": float(mu_eta[i]),
                        "phi": float(mu_phi[i]),
                        "mass": float(mu_m[i]),
                        "idx": i,
                    }

            matched_recos = []
            # Fill denominators/numerators per pT/eta bin
            for g in pair:
                aeta = abs(g["eta"])
                gpt = g["pt"]

                # find eta bin
                ie = None
                for k, (elo, ehi) in enumerate(zip(eta_edges[:-1], eta_edges[1:])):
                    if elo <= aeta < ehi:
                        ie = k
                        break
                if ie is None:
                    continue

                # find pT bin
                ip = None
                for k, (plo, phi) in enumerate(pt_bins_per_eta[ie]):
                    if plo <= gpt < phi:
                        ip = k
                        break
                if ip is None:
                    continue

                # denom
                gen_totals[ie][ip] += 1

                # numer if matched
                mrec = reco_by_gen.get(int(g["index"]))
                if mrec is not None:
                    gen_matched[ie][ip] += 1
                    matched_recos.append(mrec)

                # append momentum resolution
                if mrec is not None and gpt > 0:
                    momentum_resolutions[ie].append(
                        {
                            "gen_eta": g["eta"],
                            "reco_eta": mrec["eta"],
                            "gen_pt": gpt,
                            "reco_pt": mrec["pt"],
                            "resolution": (mrec["pt"] - gpt) / gpt,
                        }
                    )

            # Store dimuon reco mass if both legs reached reco
            if len(matched_recos) >= 2:
                matched_recos.sort(key=lambda x: x["pt"], reverse=True)
                p41 = ROOT.TLorentzVector()
                p41.SetPtEtaPhiM(
                    matched_recos[0]["pt"],
                    matched_recos[0]["eta"],
                    matched_recos[0]["phi"],
                    matched_recos[0]["mass"],
                )
                p42 = ROOT.TLorentzVector()
                p42.SetPtEtaPhiM(
                    matched_recos[1]["pt"],
                    matched_recos[1]["eta"],
                    matched_recos[1]["phi"],
                    matched_recos[1]["mass"],
                )
                mass_val = float((p41 + p42).M())

                # assign the pair to a single η slice if both gen legs are in the same slice
                aeta1 = abs(pair[0]["eta"])
                aeta2 = abs(pair[1]["eta"])
                ie1 = ie2 = None
                for k, (elo_eta, ehi_eta) in enumerate(zip(eta_edges[:-1], eta_edges[1:])):
                    if elo_eta <= aeta1 < ehi_eta:
                        ie1 = k
                    if elo_eta <= aeta2 < ehi_eta:
                        ie2 = k

                if ie1 is not None and ie1 == ie2:
                    idx = ie1
                    event_rows[idx].append(
                        {
                            "mass": mass_val,
                            "lead_pt": matched_recos[0]["pt"],
                            "lead_eta": matched_recos[0]["eta"],
                            "lead_phi": matched_recos[0]["phi"],
                            "sub_pt": matched_recos[1]["pt"],
                            "sub_eta": matched_recos[1]["eta"],
                            "sub_phi": matched_recos[1]["phi"],
                        }
                    )

        tf.Close()

    # Compute efficiencies
    eff = [[np.nan] * len(pt_bins_per_eta[ie]) for ie in range(n_eta)]
    elo = [[0.0] * len(pt_bins_per_eta[ie]) for ie in range(n_eta)]
    eup = [[0.0] * len(pt_bins_per_eta[ie]) for ie in range(n_eta)]

    for ie in range(n_eta):
        for ip in range(len(pt_bins_per_eta[ie])):
            e, l, u = clopper_pearson(gen_matched[ie][ip], gen_totals[ie][ip])
            eff[ie][ip], elo[ie][ip], eup[ie][ip] = e, l, u

    # Write CSVs
    base_prefix = f"{samp}_muon"

    # Efficiencies
    eff_path = f"{base_prefix}_efficiency.csv"
    with open(eff_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "eta_lo",
                "eta_hi",
                "pt_lo",
                "pt_hi",
                "gen_total",
                "gen_matched",
                "eff",
                "err_lo",
                "err_up",
            ]
        )
        for ie in range(n_eta):
            eta_lo = eta_edges[ie]
            eta_hi = eta_edges[ie + 1]
            for ip, (pt_lo, pt_hi) in enumerate(pt_bins_per_eta[ie]):
                n = gen_totals[ie][ip]
                k = gen_matched[ie][ip]
                e = eff[ie][ip]
                l = elo[ie][ip]
                u = eup[ie][ip]
                w.writerow(
                    [
                        f"{eta_lo:.6f}",
                        f"{eta_hi:.6f}",
                        f"{pt_lo:.6f}",
                        f"{pt_hi:.6f}",
                        n,
                        k,
                        "" if not np.isfinite(e) else f"{e:.6f}",
                        f"{l:.6f}",
                        f"{u:.6f}",
                    ]
                )
    print(f"Wrote: {eff_path}")

    # Momentum resolutions
    has_res_rows = any(momentum_resolutions[ie] for ie in range(n_eta))
    if has_res_rows:
        res_path = f"{base_prefix}_momentum_resolution.csv"
        with open(res_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "eta_lo",
                    "eta_hi",
                    "gen_eta",
                    "reco_eta",
                    "gen_pt",
                    "reco_pt",
                    "resolution",
                ]
            )
            for ie in range(n_eta):
                eta_lo = eta_edges[ie]
                eta_hi = eta_edges[ie + 1]
                for r in momentum_resolutions[ie]:
                    w.writerow(
                        [
                            f"{eta_lo:.6f}",
                            f"{eta_hi:.6f}",
                            f"{r['gen_eta']:.6f}",
                            f"{r['reco_eta']:.6f}",
                            f"{r['gen_pt']:.6f}",
                            f"{r['reco_pt']:.6f}",
                            f"{r['resolution']:.6f}",
                        ]
                    )
        print(f"Wrote: {res_path}")

    # Reco-masses
    has_kins_rows = any(event_rows[ie] for ie in range(n_eta))
    if has_kins_rows:
        kins_path = f"{base_prefix}_kins.csv"
        with open(kins_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "mass",
                    "eta_lo",
                    "eta_hi",
                    "lead_pt",
                    "lead_eta",
                    "lead_phi",
                    "sub_pt",
                    "sub_eta",
                    "sub_phi",
                ]
            )
            for ie in range(n_eta):
                eta_lo = eta_edges[ie]
                eta_hi = eta_edges[ie + 1]
                for r in event_rows[ie]:
                    w.writerow(
                        [
                            f"{r['mass']:.6f}",
                            f"{eta_lo:.6f}",
                            f"{eta_hi:.6f}",
                            f"{r['lead_pt']:.6f}",
                            f"{r['lead_eta']:.6f}",
                            f"{r['lead_phi']:.6f}",
                            f"{r['sub_pt']:.6f}",
                            f"{r['sub_eta']:.6f}",
                            f"{r['sub_phi']:.6f}",
                        ]
                    )
        print(f"Wrote: {kins_path}")

    print("Done.")


if __name__ == "__main__":
    main()
