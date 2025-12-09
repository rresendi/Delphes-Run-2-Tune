import ROOT, os, sys, math
ROOT.gROOT.SetBatch(True)

input_paths = sys.argv[1:]

# Collect root files
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

# histogram (adjust binning/range if needed)
h_mass = ROOT.TH1F(
    "h_mass",
    "Z' #rightarrow #mu#mu; m_{#mu#mu} [GeV]; Events",
    120, 2000., 6000.
)

# Loop over files
for ix, path in enumerate(files):
    print(f"[{ix+1}/{len(files)}] Opening {path}")
    tf = ROOT.TFile.Open(path, "READ")
    if not tf or tf.IsZombie():
        continue

    evs = tf.Get("Events")
    if not evs:
        tf.Close()
        continue

    # Loop events
    for iev, ev in enumerate(evs):
        nMu = getattr(ev, "nMuon", 0)
        if nMu < 2: continue

        # build local muon list
        reco = []
        for i in range(nMu):
            reco.append({
                "pt":   float(ev.Muon_pt[i]),
                "eta":  float(ev.Muon_eta[i]),
                "phi":  float(ev.Muon_phi[i]),
                "mass": float(ev.Muon_mass[i]),
                "pdgid": int(ev.Muon_pdgId[i]),
            })

        reco.sort(key=lambda m: m["pt"], reverse=True)
        mu0, mu1 = reco[0], reco[1]

        if mu0["pdgid"] * mu1["pdgid"] >= 0:
            continue

        p41 = ROOT.TLorentzVector()
        p41.SetPtEtaPhiM(mu0["pt"], mu0["eta"], mu0["phi"], mu0["mass"])
        p42 = ROOT.TLorentzVector()
        p42.SetPtEtaPhiM(mu1["pt"], mu1["eta"], mu1["phi"], mu1["mass"])
        m = (p41 + p42).M()
        h_mass.Fill(m)
        ibin = h_mass.GetMaximumBin()
        mode = h_mass.GetBinCenter(ibin)
        count = h_mass.GetBinContent(ibin)
    tf.Close()
    print("mode (bin center) =", mode)
    print("entries in mode bin =", count)

# Draw and save histogram
c = ROOT.TCanvas("c", "c", 800, 600)
h_mass.GetXaxis().SetRangeUser(4000, 6000)
h_mass.SetLineWidth(2)
h_mass.Draw("HIST")
c.SaveAs("zprime_mass.pdf")

print("Saved mass distribution → zprime_mass.pdf")
