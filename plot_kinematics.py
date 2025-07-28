import ROOT
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep

def main():
    input_dir = "/eos/user/g/gdecastr/sampleFactoryOutput/SampleFactory/JPsi_DiMu_Pythia8/chain_SMP-RunIISummer20UL18GEN-00050-SMP-RunIISummer20UL18NanoAODv9-00368/20250726_062214/"
    inputFiles = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".root")]

    inF = 0
    iEv = 0
    nF = len(inputFiles)

    # Output lists
    lead_pts, sub_pts = [], []
    lead_etas, sub_etas = [], []
    lead_phis, sub_phis = [], []
    masses = []

    # Start looping through files in the directory
    for i, iFile in enumerate(inputFiles):
        inF += 1
        # print(f"Processing file {inF}/{nF}: {iFile}")

        # Open and read the ROOT file
        tf = ROOT.TFile.Open(iFile, "READ")
        if not tf or tf.IsZombie():
            print(f"Error opening file {iFile}. Skipping.")
            continue

        events = tf.Get("Events")
        if not events:
            print(f"No 'Events' tree found in {iFile}. Skipping.")
            tf.Close()
            continue
        
        # Start looping through the events in a given file
        for ev in events:

            # Extract relevant branches
            nGenPart = ev.nGenPart
            pdgIds = ev.GenPart_pdgId
            pts = ev.GenPart_pt
            etas = ev.GenPart_eta
            phis = ev.GenPart_phi
            masses_branch = ev.GenPart_mass

            # Create muon collection
            muons = []
            for i in range(nGenPart):
                # Apply object-level selections
                if abs(pdgIds[i]) == 13 and abs(etas[i]) < 2.4:
                    muons.append({
                        "pt": pts[i],
                        "eta": etas[i],
                        "phi": phis[i],
                        "mass": masses_branch[i],
                        "pdgId": pdgIds[i]
                    })

            # Sort by pt
            muons = sorted(muons, key=lambda x: x["pt"], reverse=True)

            if len(muons) < 2:
                continue

            # Take the leading lepton
            mu1 = muons[0]

            # Loop through subleading leptons to create OSSF pairs
            for mu2 in muons[1:]:
                if mu1["pdgId"] * mu2["pdgId"] >= 0:
                    continue

                # Save kinematics
                lead_pts.append(mu1["pt"])
                lead_etas.append(mu1["eta"])
                lead_phis.append(mu1["phi"])

                sub_pts.append(mu2["pt"])
                sub_etas.append(mu2["eta"])
                sub_phis.append(mu2["phi"])

                # Invariant mass
                p4_1 = ROOT.TLorentzVector()
                p4_1.SetPtEtaPhiM(mu1["pt"], mu1["eta"], mu1["phi"], mu1["mass"])
                p4_2 = ROOT.TLorentzVector()
                p4_2.SetPtEtaPhiM(mu2["pt"], mu2["eta"], mu2["phi"], mu2["mass"])
                inv_mass = (p4_1 + p4_2).M()
                masses.append(inv_mass)

            iEv += 1

    print(f"Finished processing {inF} files with {iEv} events.")
    print(f"Total OSSF pairs found: {len(masses)}")

    # Plotting

    # Dilepton mass
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(8, 8))
    hep.style.use("CMS")
    hep.cms.label(data = True, label="Preliminary")
    plt.hist(masses, bins=50, range=(3, 3.3), histtype='step', color=cmap(0), linewidth=2)
    plt.xlabel("Dilepton Mass [GeV]")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig("dilepton_mass.png")
    plt.close()

    # pT
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(8, 8))
    hep.style.use("CMS")
    hep.cms.label(data = True, label="Preliminary")
    plt.hist(lead_pts, bins=20, range=(0, 10), histtype='step', label="Leading", color=cmap(0), linewidth=2)
    plt.hist(sub_pts, bins=20, range=(0, 10), histtype='step', label="Subleading", color=cmap(2), linewidth=2)
    plt.xlabel("Muon $p_T$ [GeV]")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig("muon_pt.png")
    plt.close()

    # Eta
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(8, 8))
    hep.style.use("CMS")
    hep.cms.label(data = True, label="Preliminary")
    plt.hist(lead_etas, bins=20, range=(-3, 3), histtype='step', label="Leading", color=cmap(0), linewidth=2)
    plt.hist(sub_etas, bins=20, range=(-3, 3), histtype='step', label="Subleading", color=cmap(2), linewidth=2)
    plt.xlabel("Muon η")
    plt.ylabel("Counts")
    plt.legend()
    # plt.tight_layout()
    plt.savefig("muon_eta.png")
    plt.close()

    # Phi
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(8, 8))
    hep.style.use("CMS")
    hep.cms.label(data = True, label="Preliminary")
    plt.hist(lead_phis, bins=20, range=(-np.pi, np.pi), histtype='step', label="Leading", color=cmap(0), linewidth=2)
    plt.hist(sub_phis, bins=20, range=(-np.pi, np.pi), histtype='step', label="Subleading", color=cmap(2), linewidth=2)
    plt.xlabel("Muon φ")
    plt.ylabel("Counts")
    plt.legend()
    # plt.tight_layout()
    plt.savefig("muon_phi.png")
    plt.close()

    print("Saved all plots.")

if __name__ == "__main__":
    main()
