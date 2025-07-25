import uproot
import ROOT
import awkward as ak
import sys
import numpy as np
import argparse
import array

# Initialize argparse
parser = argparse.ArgumentParser()
parser.add_argument("--obj", choices=["electron", "muon", "pion"])
parser.add_argument("--sample", choices=["onshell", "offshell", "piguns"])
args = parser.parse_args()

obj = args.obj
sample = args.sample

# Open the ROOT file
file = uproot.open("root_file.root")
events = file["Events"]

# Define object groups at gen, PF, and reco level
gen = events.arrays(["GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_pdgId"])

# ******* don't know the semantics to access pfcand info so this is a placeholder for now *******
pf = events.arrays(["PFCands_eta", "PFCands_phi", "PFCands_charge", "PFCands_pt"])

# DeltaR function
def deltaR(eta1, phi1, eta2, phi2):
  dphi = phi1 - phi2
  dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
  deta = eta1 - eta2
  deltaR = np.sqrt(deta ** 2 + dphi **2)
  return deltaR

# Invariant mass function
def invariant_mass(pt1, eta1, phi1, pt2, eta2, phi2):
    # Four momentum components
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = pt1 * np.cosh(eta1)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = pt2 * np.cosh(eta2)

    # Sum four vectors
    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    return np.sqrt(e**2 - px**2 - py**2 - pz**2)

# Select charged PF candidates
is_charged_pf = (pf["PFCands_charge"] != 0)
pf_eta = pf["PFCands_eta"][is_charged_pf]
pf_phi = pf["PFCands_phi"][is_charged_pf]
pf_pt = pf["PFCands_pt"][is_charged_pf]

if obj == "electron":
  # Select gen level electrons
  is_gen = (abs(gen["GenPart_pdgId"]) == 11)

  # Select reco level electrons
  reco = events.arrays(["Electron_eta", "Electron_phi", "Electron_pt"])
  reco_eta = reco["Electron_eta"]
  reco_phi = reco["Electron_phi"]
  reco_pt = reco["Electron_pt"]

elif obj == "pion":
  # Select gen level pions
  is_gen = (abs(gen["GenPart_pdgId"]) == 211)

  # No reco level info needed
  reco_eta = None
  reco_phi = None
  reco_pt = None

  # Check for at least one jet in the event
  jets = events.arrays(["Jet_pt"])
  if ak.count(jets["Jet_pt"], axis=1).max() <= 1:
      print("No events with >1 jet found.")
      sys.exit(0)
  multi_jet_mask = ak.count(jets["Jet_pt"], axis=1) > 1

elif obj == "muon":
  # Select gen level muons
  is_gen = (abs(gen["GenPart_pdgId"]) == 13)

  # Select reco level electrons
  reco = events.arrays(["Muon_eta", "Muon_phi", "Muon_pt"])
  reco_eta = reco["Muon_eta"]
  reco_phi = reco["Muon_phi"]
  reco_pt = reco["Muon_pt"]

# Skip events with no gen particles
gen_pt = gen["GenPart_pt"][is_gen]
if ak.count(gen_pt) == 0:
    print("Skipping event. No matching gen particles found.")
    sys.exit(0)

# Apply gen particle and eta selection
gen_eta = gen["GenPart_eta"][is_gen]
gen_phi = gen["GenPart_phi"][is_gen]
gen_id = gen["GenPart_pdgId"][is_gen]
eta_mask = abs(gen_eta) < 2.4
gen_eta = gen_eta[eta_mask]
gen_phi = gen_phi[eta_mask]
gen_pt = gen_pt[eta_mask]
gen_id = gen_id[eta_mask]

# Apply nJets > 1 selection for pions
if obj == "pion":
    gen_eta = gen_eta[multi_jet_mask]
    gen_phi = gen_phi[multi_jet_mask]
    gen_pt = gen_pt[multi_jet_mask]

# Apply OSSF and onshell selections for leptons
if obj in ["electron", "muon"] and sample in ["onshell", "offshell"]:
  # Create all possible gen-gen lepton pairs
  gen_leptons = ak.zip({
      "eta": gen_eta,
      "phi": gen_phi,
      "pt": gen_pt,
      "pdgId": gen_id
  })

  gen_pairs = ak.cartesian([gen_leptons, gen_leptons], nested=True)
  lep1 = gen_pairs["0"]
  lep2 = gen_pairs["1"]

  # Apply OSSF condition using pdgId
  is_ossf = (lep1.pdgId + lep2.pdgId == 0)

  # Avoid self-pairing
  not_same = lep1.pt != lep2.pt

  # Compute invariant mass for all pairs
  mass = invariant_mass(lep1.pt, lep1.eta, lep1.phi, lep2.pt, lep2.eta, lep2.phi)

  # Apply mass window
  if sample == "onshell":
      mass_cut = (mass > 81) & (mass < 120)
  elif sample == "offshell":
      mass_cut = (mass > 2) & (mass < 20)
  else:
      mass_cut = ak.ones_like(mass, dtype=bool)  # No cut for piguns or other samples

  # Combine masks
  valid_pairs = is_ossf & not_same & mass_cut

  # Find events that contain at least one valid OSSF pair
  ossf_mask = ak.any(valid_pairs, axis=1)

  # Apply the mask to gen leptons and everything else downstream
  gen_eta = gen_eta[ossf_mask]
  gen_phi = gen_phi[ossf_mask]
  gen_pt  = gen_pt[ossf_mask]
  pf_eta  = pf_eta[ossf_mask]
  pf_phi  = pf_phi[ossf_mask]
  pf_pt   = pf_pt[ossf_mask]

# Set up efficiency histograms
ROOT.gROOT.SetBatch(True)

# Variable pT binning
pt_bins = np.array([
0, 2, 4, 6, 8, 10, 12,
14, 16, 18, 20, 22,
24, 26, 28, 30, 32,
34, 36, 38, 40, 50,
60, 70, 80, 90, 100,
120, 140, 160, 180, 200
], dtype = float)

# Create efficiency histograms
track_eff_histo = ROOT.TEfficiency("track_eff_histo", f"Track Efficiency;Gen pT [GeV];Efficiency", len(pt_bins)-1, pt_bins)
calo_eff_histo = ROOT.TEfficiency("calo_eff_histo", f"Calo/Chamber Efficiency;Gen pT [GeV];Efficiency", len(pt_bins)-1, pt_bins)

# Eta-binned for electrons only
if obj == "electron":
    eta_bins = [(0.0, 0.8), (0.8, 1.44), (1.57, 2.4)]

    track_eta_eff_histos = []
    calo_eta_eff_histos = []

    for i, (eta_min, eta_max) in enumerate(eta_bins):
        track_hist = ROOT.TEfficiency(
            f"track_eff_eta_bin_{i}",
            f"Track Efficiency;Gen pT [GeV];Efficiency;Eta [{eta_min}, {eta_max}]",
            len(pt_bins) - 1, pt_bins
        )
        calo_hist = ROOT.TEfficiency(
            f"calo_eff_eta_bin_{i}",
            f"Calo/Chamber Efficiency;Gen pT [GeV];Efficiency;Eta [{eta_min}, {eta_max}]",
            len(pt_bins) - 1, pt_bins
        )
        track_eta_eff_histos.append(track_hist)
        calo_eta_eff_histos.append(calo_hist)

# Create gen and PF eta/phi object pairs
gen_objs = ak.zip({"eta": gen_eta, "phi": gen_phi, "pt": gen_pt})
pf_objs = ak.zip({"eta": pf_eta, "phi": pf_phi, "pt": pf_pt})
gen_pf_pairs = ak.cartesian([gen_objs, pf_objs], nested=True)

# Calculate deltaR for these pairs
gen_pf_dR = deltaR(
gen_pf_pairs["0"].eta, gen_pf_pairs["0"].phi,
gen_pf_pairs["1"].eta, gen_pf_pairs["1"].phi,
)

# Use best match only
best_pf_id = ak.argmin(gen_pf_dR, axis=1)
best_pf_dR = ak.firsts(ak.sort(gen_pf_dR, axis=1))

# Apply matching conditions
gen_pf_dr_match = best_pf_dR < 0.1
pt_ratio = pf_objs.pt[best_pf_id] / gen_pt
pt_match = (pt_ratio > 0.7) & (pt_ratio < 1.3)

gen_pf_match = gen_pf_dr_match & pt_match

# Calculate efficiency at the tracker level
gen_matched_to_pf = ak.any(gen_pf_match, axis=1)
pf_matched_mask = ak.any(gen_pf_match, axis=0)

# Fill track efficiency histogram for muons and pions
if obj not in ["electron"]:
  for pt, matched in zip(ak.flatten(gen_pt), ak.flatten(gen_matched_to_pf)):
    track_eff_histo.Fill(bool(matched), pt)

# Fill eta-binned track efficiency histogram for electrons
elif obj == "electron":
    for i, (eta_min, eta_max) in enumerate(eta_bins):
        eta_mask = (abs(gen_eta) >= eta_min) & (abs(gen_eta) < eta_max)

        # Track efficiency filling
        gen_pt_bin = gen_pt[eta_mask]
        matched_track_bin = gen_matched_to_pf[eta_mask]

        for pt, matched in zip(ak.flatten(gen_pt_bin), ak.flatten(matched_track_bin)):
            track_eta_eff_histos[i].Fill(bool(matched), pt)

# Calculate efficiency at the calorimeter/muon chamber level for leptons only
if obj in ["electron", "muon"]:

  # Find PFs matched to Gen
  pf_matched = ak.any(gen_pf_match, axis=0)
  matched_pf_eta = pf_eta[pf_matched]
  matched_pf_phi = pf_phi[pf_matched]
  matched_pf_pt = pf_pt[pf_matched]

  # Create PF and reco eta/phi object pairs
  matched_pf_for_reco = ak.zip({"eta": matched_pf_eta, "phi": matched_pf_phi, "pt": matched_pf_pt})
  reco = ak.zip({"eta": reco_eta, "phi": reco_phi, "pt": reco_pt})

  # Build PF–Reco candidate pairs
  pf_reco_pairs = ak.cartesian([matched_pf_for_reco, reco], nested=True)

  # Calculate deltaR of these pairs
  pf_reco_dR = deltaR(
      pf_reco_pairs["0"].eta, pf_reco_pairs["0"].phi,
      pf_reco_pairs["1"].eta, pf_reco_pairs["1"].phi,
  )

  # Get best PF match index per gen
  best_pf_id = ak.argmin(gen_pf_dR, axis=1)
  matched_mask = gen_pf_match

  # Only keep gen leptons that were matched
  gen_matched_mask = ak.any(matched_mask, axis=1)
  matched_pf_id = best_pf_id[gen_matched_mask]

  # Get PF kinematics from matched PFs
  matched_pf_eta = pf_eta[matched_pf_id]
  matched_pf_phi = pf_phi[matched_pf_id]
  matched_pf_pt = pf_pt[matched_pf_id]

  # Rebuild the PF objects to match Reco
  matched_pf_for_reco = ak.zip({"eta": matched_pf_eta, "phi": matched_pf_phi, "pt": matched_pf_pt})
  reco = ak.zip({"eta": reco_eta, "phi": reco_phi, "pt": reco_pt})

  pf_reco_dr_match = pf_reco_dR < 0.1
  pt_ratio = pf_reco_pairs["1"].pt / pf_reco_pairs["0"].pt
  pt_match = (pt_ratio > 0.7) & (pt_ratio < 1.3)
  pf_reco_match = pf_reco_dr_match & pt_match

  # Calculate Muon Chamber efficiency
  pf_matched_to_reco = ak.any(pf_reco_match, axis=1)
  matched_gen_pt = gen_pt[gen_matched_mask]
  if obj not in ["electron"]:
    for pt, matched in zip(ak.flatten(matched_gen_pt), ak.flatten(pf_matched_to_reco)):
        calo_eff_histo.Fill(bool(matched), pt)

  # Calculate eta-binned ECAL efficiency
  if obj == "electron":
    matched_gen_eta = gen_eta[gen_matched_mask]

    for i, (eta_min, eta_max) in enumerate(eta_bins):
        eta_mask = (abs(matched_gen_eta) >= eta_min) & (abs(matched_gen_eta) < eta_max)

        matched_gen_pt_bin = matched_gen_pt[eta_mask]
        matched_calo_bin = pf_matched_to_reco[eta_mask]

        for pt, matched in zip(ak.flatten(matched_gen_pt_bin), ak.flatten(matched_calo_bin)):
            calo_eta_eff_histos[i].Fill(bool(matched), pt)
  
outfile = ROOT.TFile(f"{obj}_{sample}_track_eff.root", "RECREATE")
track_eff_histo.Write()
calo_eff_histo.Write()
if obj == "electron":
    for hist in track_eta_eff_histos:
        hist.Write()
    for hist in calo_eta_eff_histos:
        hist.Write()
outfile.Close()

print(f"Efficiencies for {obj} in {sample} written to {obj}_{sample}_track_eff.root")
