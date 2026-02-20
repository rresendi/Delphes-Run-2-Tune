#!/usr/bin/env python3
import os, sys, csv, math
import ROOT
import numpy as np

ROOT.gROOT.SetBatch(True)

CL = 0.683
ALPHA = 1.0 - CL

def cp(k, n):
    if n <= 0:
        return float("nan"), 0.0, 0.0
    eff = k / n
    lo = ROOT.TEfficiency.ClopperPearson(int(n), int(k), ALPHA, False)
    hi = ROOT.TEfficiency.ClopperPearson(int(n), int(k), ALPHA, True)
    return eff, eff - lo, hi - eff

def dphi(a, b):
    x = a - b
    while x > math.pi:
        x -= 2 * math.pi
    while x <= -math.pi:
        x += 2 * math.pi
    return x

def dR(eta1, phi1, eta2, phi2):
    return math.hypot(eta1 - eta2, dphi(phi1, phi2))

def files_from(path):
    if os.path.isfile(path) and path.endswith(".root"):
        return [path]
    if os.path.isfile(path) and path.endswith(".txt"):
        with open(path) as f:
            return [l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")]
    if os.path.isdir(path):
        return [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".root")]
    return []

def find_bin(x, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return i
    return None

def best_pair(gen, samp, m0):
    mass_wins = {"jpsi": (2.9, 3.3), "dy": (60, 120), "zprime": (2000, 5000)}
    lo, hi = mass_wins[samp]
    best, best_dm = None, 1e18
    for i, a in enumerate(gen):
        for b in gen[i + 1 :]:
            if a["pdgId"] * b["pdgId"] >= 0:
                continue
            p1 = ROOT.TLorentzVector(); p1.SetPtEtaPhiM(a["pt"], a["eta"], a["phi"], a["m"])
            p2 = ROOT.TLorentzVector(); p2.SetPtEtaPhiM(b["pt"], b["eta"], b["phi"], b["m"])
            m = (p1 + p2).M()
            if not (lo <= m <= hi):
                continue
            dm = abs(m - m0)
            if dm < best_dm:
                best_dm = dm
                best = (a, b)
    return best

# Gen to PF matching
def match_pftrack(g, trk_eta, trk_phi, dr_max=0.1):
    geta, gphi = float(g["eta"]), float(g["phi"])
    best = None
    best_dr = 1e9
    n = min(len(trk_eta), len(trk_phi))
    for i in range(n):
        eta = float(trk_eta[i])
        phi = float(trk_phi[i])

        # crack veto
        if 1.444 < abs(eta) < 1.566:
            continue

        dr = dR(geta, gphi, eta, phi)
        if dr >= dr_max:
            continue
        if dr < best_dr:
            best_dr = dr
            best = {"eta": eta, "phi": phi, "dr": dr, "i": i}
    return best

# (passing) Gen to Reco matching
def match_reco(g, reco_list, used_reco_idx, dr_max=0.1):
    geta, gphi = float(g["eta"]), float(g["phi"])
    best = None
    best_dr = 1e9
    for r in reco_list:
        if r["i"] in used_reco_idx:
            continue
        dr = dR(geta, gphi, float(r["eta"]), float(r["phi"]))
        if dr >= dr_max:
            continue
        if dr < best_dr:
            best_dr = dr
            best = r
    return best

def write_eff_csv(path, eta_edges, pt_bins, tot, hit):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eta_lo","eta_hi","pt_lo","pt_hi","gen_total","gen_matched","eff","err_lo","err_up"])
        for ie in range(len(eta_edges) - 1):
            for ip, (plo, phi) in enumerate(pt_bins):
                n, k = tot[ie][ip], hit[ie][ip]
                e, el, eu = cp(k, n)
                w.writerow([
                    f"{eta_edges[ie]:.6f}", f"{eta_edges[ie+1]:.6f}",
                    f"{plo:.6f}", f"{phi:.6f}",
                    n, k,
                    "" if not np.isfinite(e) else f"{e:.6f}",
                    f"{el:.6f}", f"{eu:.6f}",
                ])

def main():

    samp = sys.argv[1]  # jpsi, dy, or zprime
    files = []
    for p in sys.argv[2:]:
        files += files_from(p)

    eta_edges = [0.0, 0.8, 1.444, 1.98, 2.5]

    if samp == "jpsi":
        pt_edges, m0, pt_sel = [1, 2, 3, 4, 5, 6, 7, 8, 10], 3.0969, (1.0, 10.0)
    elif samp == "dy":
        pt_edges, m0, pt_sel = [10.0, 100.0, 1000.0], 91.1876, (10.0, 1000.0)
    else:
        pt_edges, m0, pt_sel = [1000.0, 3000.0], 5000.0, (1000.0, 3000.0)

    pt_bins = list(zip(pt_edges[:-1], pt_edges[1:]))
    n_eta, n_pt = len(eta_edges) - 1, len(pt_bins)

    tot1 = [[0] * n_pt for _ in range(n_eta)]
    hit1 = [[0] * n_pt for _ in range(n_eta)]
    res_rows = []

    tot2 = [[0] * n_pt for _ in range(n_eta)]
    hit2 = [[0] * n_pt for _ in range(n_eta)]
    mass_rows = []

    # matching thresholds
    DR_PF = 0.1
    DR_RECO = 0.1

    for iz, fpath in enumerate(files):
        print(f"[{iz + 1}/{len(files)}] {fpath}")
        tf = ROOT.TFile.Open(fpath)
        t = tf.Get("Events")
        if not t:
            tf.Close()
            continue

        for ev in t:
            # Build gen collection
            gp_pdg = getattr(ev, "GenPart_pdgId", None)
            gp_pt  = getattr(ev, "GenPart_pt", None)
            gp_eta = getattr(ev, "GenPart_eta", None)
            gp_phi = getattr(ev, "GenPart_phi", None)
            gp_m   = getattr(ev, "GenPart_mass", None)
            gp_st  = getattr(ev, "GenPart_status", None)
            nGen   = int(getattr(ev, "nGenPart", 0))
            if any(x is None for x in (gp_pdg, gp_pt, gp_eta, gp_phi, gp_m, gp_st)):
                continue

            gen = []
            for i in range(nGen):
                if abs(int(gp_pdg[i])) != 11:
                    continue
                if int(gp_st[i]) != 1:
                    continue
                eta = float(gp_eta[i])
                if 1.444 <= abs(eta) <= 1.566:
                    continue
                if abs(eta) >= 2.5:
                    continue
                pt = float(gp_pt[i])
                if not (pt_sel[0] <= pt < pt_sel[1]):
                    continue
                gen.append({"idx": i, "pt": pt, "eta": eta, "phi": float(gp_phi[i]), "m": float(gp_m[i]), "pdgId": int(gp_pdg[i])})

            if len(gen) < 2:
                continue

            gen.sort(key=lambda x: x["pt"], reverse=True)
            pair = best_pair(gen, samp, m0)
            if not pair:
                continue
            pair = [pair[0], pair[1]]

            # Build PF track collection
            pf_trkEta = getattr(ev, "PFCands_trkEta", None)
            pf_trkPhi = getattr(ev, "PFCands_trkPhi", None)
            pf_trkPt  = getattr(ev, "PFCands_trkPt", None)
            pf_pdgId  = getattr(ev, "PFCands_pdgId", None)

            if any(x is None for x in (pf_trkEta, pf_trkPhi, pf_trkPt, pf_pdgId)):
                continue

            nPF = min(len(pf_trkEta), len(pf_trkPhi), len(pf_trkPt), len(pf_pdgId))
            if nPF == 0:
                continue

            # Veto PF muons
            has_pf_muon = False
            for i in range(nPF):
                if abs(int(pf_pdgId[i])) == 13:
                    has_pf_muon = True
                    break
            if has_pf_muon:
                continue

            # Build reco electron collection
            nEle = int(getattr(ev, "nElectron", 0))
            ele_pt   = getattr(ev, "Electron_pt", None)
            ele_eta  = getattr(ev, "Electron_eta", None)
            ele_phi  = getattr(ev, "Electron_phi", None)
            ele_m    = getattr(ev, "Electron_mass", None)
            ele_med  = getattr(ev, "Electron_cutBased", None)

            reco_selected = []
            if not any(x is None for x in (ele_pt, ele_eta, ele_phi, ele_m, ele_med)):
                for i in range(nEle):
                    eta = float(ele_eta[i])
                    if 1.444 <= abs(eta) <= 1.566:
                        continue
                    if abs(eta) >= 2.5:
                        continue
                    # turn id on/off
                    # if int(ele_med[i]) < 3:
                    #     continue
                    reco_selected.append({"i": i, "pt": float(ele_pt[i]), "eta": eta, "phi": float(ele_phi[i]), "m": float(ele_m[i])})

            # Calculate first efficiency
            passed1 = {}
            for g in pair:
                ie = find_bin(abs(g["eta"]), eta_edges)
                ip = next((j for j, (plo, phi) in enumerate(pt_bins) if plo <= g["pt"] < phi), None)
                if ie is None or ip is None:
                    continue

                tot1[ie][ip] += 1

                m = match_pftrack(g, pf_trkEta, pf_trkPhi, dr_max=DR_PF)
                if not m:
                    continue

                hit1[ie][ip] += 1
                passed1[int(g["idx"])] = (ie, ip)

                # write info for momentum resolution csv
                ti = int(m["i"])
                res_rows.append({
                    "pf_pt": float(pf_trkPt[ti]),
                    "gen_pt": float(g["pt"]),
                    "pf_eta": float(pf_trkEta[ti]),
                    "gen_eta": float(g["eta"]),
                    "resolution": (float(pf_trkPt[ti]) - float(g["pt"])) / float(g["pt"]) if float(g["pt"]) > 0 else float("nan"),
                })

            # Calculate reco efficiency using passing gen objs from track eff
            reco_legs = []
            used_reco_idx = set()

            for g in pair:
                gidx = int(g["idx"])
                if gidx not in passed1:
                    continue

                ie, ip = passed1[gidx]
                tot2[ie][ip] += 1

                best_reco = match_reco(g, reco_selected, used_reco_idx, dr_max=DR_RECO)
                if best_reco is None:
                    continue

                hit2[ie][ip] += 1
                used_reco_idx.add(best_reco["i"])
                reco_legs.append(best_reco)

            # Calculate inv. mass if both efficiencies are passed
            if len(passed1) == 2 and len(reco_legs) == 2:
                reco_legs.sort(key=lambda x: x["pt"], reverse=True)
                p1 = ROOT.TLorentzVector(); p1.SetPtEtaPhiM(reco_legs[0]["pt"], reco_legs[0]["eta"], reco_legs[0]["phi"], reco_legs[0]["m"])
                p2 = ROOT.TLorentzVector(); p2.SetPtEtaPhiM(reco_legs[1]["pt"], reco_legs[1]["eta"], reco_legs[1]["phi"], reco_legs[1]["m"])
                mass_rows.append({
                    "mass": float((p1 + p2).M()),
                    "lead_pt": reco_legs[0]["pt"],
                    "lead_eta": reco_legs[0]["eta"],
                    "lead_phi": reco_legs[0]["phi"],
                    "sub_pt": reco_legs[1]["pt"],
                    "sub_eta": reco_legs[1]["eta"],
                    "sub_phi": reco_legs[1]["phi"],
                })

        tf.Close()

    outbase = f"ele_{samp}_pftrk_pfveto"

    eff1_path = f"{outbase}_track_efficiency_pfcands.csv"
    write_eff_csv(eff1_path, eta_edges, pt_bins, tot1, hit1)
    print(f"Wrote: {eff1_path}")

    res_path = f"{outbase}_momentum_resolution_pfcands.csv"
    with open(res_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pf_pt","gen_pt","pf_eta","gen_eta","resolution"])
        for r in res_rows:
            w.writerow([f"{r['pf_pt']:.6f}", f"{r['gen_pt']:.6f}", f"{r['pf_eta']:.6f}", f"{r['gen_eta']:.6f}", f"{r['resolution']:.6f}"])
    print(f"Wrote: {res_path}")

    eff2_path = f"{outbase}_reconstruction_efficiency_pfcands.csv"
    write_eff_csv(eff2_path, eta_edges, pt_bins, tot2, hit2)
    print(f"Wrote: {eff2_path}")

    mass_path = f"{outbase}_reco_mass_pfcands.csv"
    with open(mass_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mass","lead_pt","lead_eta","lead_phi","sub_pt","sub_eta","sub_phi"])
        for r in mass_rows:
            w.writerow([
                f"{r['mass']:.6f}",
                f"{r['lead_pt']:.6f}", f"{r['lead_eta']:.6f}", f"{r['lead_phi']:.6f}",
                f"{r['sub_pt']:.6f}",  f"{r['sub_eta']:.6f}",  f"{r['sub_phi']:.6f}",
            ])
    print(f"Wrote: {mass_path}")

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
