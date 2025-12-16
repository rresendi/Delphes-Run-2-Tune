#!/usr/bin/env python3
import os, sys
import ROOT
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

ROOT.gROOT.SetBatch(True)

input_paths = sys.argv[1:]
files = []

for p in input_paths:
    if os.path.isdir(p):
        for fn in os.listdir(p):
            if fn.endswith(".root"):
                files.append(os.path.join(p, fn))
    elif p.endswith(".root"):
        files.append(p)

if not files:
    print("No ROOT files found.")
    sys.exit(1)

dimuon_masses = []

for ix, path in enumerate(files):
    print(f"[{ix+1}/{len(files)}] Opening {path}")
    tf = ROOT.TFile.Open(path)
    if not tf or tf.IsZombie():
        continue

    evs = tf.Get("Events")
    if not evs:
        tf.Close()
        continue

    for ev in evs:
        if ev.nMuon < 2:
            continue

        muons = []
        for i in range(ev.nMuon):
            muons.append({
                "pt":   float(ev.Muon_pt[i]),
                "eta":  float(ev.Muon_eta[i]),
                "phi":  float(ev.Muon_phi[i]),
                "mass": float(ev.Muon_mass[i]),
                "pdg":  int(ev.Muon_pdgId[i]),
            })

        muons.sort(key=lambda m: m["pt"], reverse=True)
        mu0, mu1 = muons[0], muons[1]

        # opposite sign
        if mu0["pdg"] * mu1["pdg"] >= 0:
            continue

        p40 = ROOT.TLorentzVector()
        p41 = ROOT.TLorentzVector()
        p40.SetPtEtaPhiM(mu0["pt"], mu0["eta"], mu0["phi"], mu0["mass"])
        p41.SetPtEtaPhiM(mu1["pt"], mu1["eta"], mu1["phi"], mu1["mass"])

        dimuon_masses.append((p40 + p41).M())

    tf.Close()

print(f"Collected {len(dimuon_masses)} dimuon candidates")
masses = np.array(dimuon_masses)

hep.style.use("CMS")
plt.figure(figsize=(8, 8))
hep.cms.label(data=True, label="Preliminary")

plt.hist(
    masses,
    bins=120,
    range=(2000, 6000),
    histtype="step",
    linewidth=2,
    color="green"
)

plt.xlabel(r"$m_{\mu\mu}$ [GeV]")
plt.ylabel("Events")
plt.xlim(2000, 6000)

plt.tight_layout()
plt.savefig("zp_mass.pdf")
plt.close()

print("Saved plot → zp_mass.pdf")
