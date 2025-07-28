import ROOT
import os
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep

def main():
    input_dir = "/eos/user/r/rresendi/DYtest/"
    inputFiles = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".root")]

    inF = 0
    iEv = 0
    nF = len(inputFiles)

    # Output lists
    lead_pts, sub_pts, gen_pts = [], [], []
    lead_etas, sub_etas, gen_etas = [], [], []
    lead_phis, sub_phis, gen_phis = [], [], []
    masses = []
    matched_gen_pts = []

    # Initialize counts
    n_matched = 0

    # DeltaR helper function
    def deltaR(eta1, phi1, eta2, phi2):
        dphi = np.abs(phi1 - phi2)
        if dphi > np.pi:
            dphi = 2*np.pi - dphi
        deta = eta1 - eta2
        return np.sqrt(deta**2 + dphi**2)

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

        # Initialize numerator and denominator for the tracking efficiency
        n_gen_total = 0
        n_matched_total = 0
        
        # Start looping through the events in a given file
        for ev in events:
            # Initialize tracking efficiency denominator
            track_eff_denominator_muons = set()

            # Initialize n_matched for each event
            n_matched = 0

            # Extract relevant branches
            nGenPart = ev.nGenPart
            pdgIds = ev.GenPart_pdgId
            pts = ev.GenPart_pt
            etas = ev.GenPart_eta
            phis = ev.GenPart_phi
            masses_branch = ev.GenPart_mass

            # Create gen muon collection
            gen_muons = []
            for i in range(nGenPart):
                # Apply object-level selections
                if abs(pdgIds[i]) == 13 and abs(etas[i]) < 2.4:
                    gen_muons.append({
                        "pt": pts[i],
                        "eta": etas[i],
                        "phi": phis[i],
                        "mass": masses_branch[i],
                        "pdgId": pdgIds[i]
                    })

            # Sort by pt
            gen_muons = sorted(gen_muons, key=lambda x: x["pt"], reverse=True)

            # Create gen muon list once per event
            for gm in gen_muons:
                gen_pts.append(gm["pt"])
                gen_etas.append(gm["eta"])
                gen_phis.append(gm["phi"])

            if len(gen_muons) < 2:
                continue

            # Initialize PF Candidate list
            pf_candidates = []

            # Loop through subleading leptons to create OSSF pairs
            denominator_muons = set()
            for i, mu1 in enumerate(gen_muons):
                # Do the OSSF check
                for mu2 in gen_muons[i+1:]:
                    if mu1["pdgId"] * mu2["pdgId"] >= 0:
                        continue

                    # Calculate dilepton invariant mass only using OSSF pairs
                    p4_1 = ROOT.TLorentzVector()
                    p4_1.SetPtEtaPhiM(mu1["pt"], mu1["eta"], mu1["phi"], mu1["mass"])
                    p4_2 = ROOT.TLorentzVector()
                    p4_2.SetPtEtaPhiM(mu2["pt"], mu2["eta"], mu2["phi"], mu2["mass"])

                    # Event level selection that only takes pairs on the j/psi resonance
                    inv_mass = (p4_1 + p4_2).M()
                    if inv_mass < 2.9 or inv_mass > 3.3:
                        continue

                    # Appending to lists for plotting
                    masses.append(inv_mass)
                    lead_pts.append(mu1["pt"])
                    lead_etas.append(mu1["eta"])
                    lead_phis.append(mu1["phi"])
                    sub_pts.append(mu2["pt"])
                    sub_etas.append(mu2["eta"])
                    sub_phis.append(mu2["phi"])

                    # Adding muons to the OSSF gen-muons set
                    track_eff_denominator_muons.add((mu1["pt"], mu1["eta"], mu1["phi"]))
                    track_eff_denominator_muons.add((mu2["pt"], mu2["eta"], mu2["phi"]))

            # Build gen-level muon kinematic lists
            track_eff_denominator_pts = [pt for pt, eta, phi in track_eff_denominator_muons]
            track_eff_denominator_etas = [eta for pt, eta, phi in track_eff_denominator_muons]
            track_eff_denominator_phis = [phi for pt, eta, phi in track_eff_denominator_muons]

            # Initialize a list of PF candidates
            for i in range(ev.nPFCands):
                if abs(ev.PFCands_pdgId[i]) == 13:
                    pf_candidates.append({
                    "index": i,
                    "pt": ev.PFCands_pt[i],
                    "eta": ev.PFCands_eta[i],
                    "phi": ev.PFCands_phi[i]
                    })

            # Loop over all gen muons and try to match to a PF candidate
            for ig, (gpt, geta, gphi) in enumerate(zip(track_eff_denominator_pts, track_eff_denominator_etas, track_eff_denominator_phis)):

                best_match = None

                # Initialize best deltaR and PtRel to an unphysical value
                best_dR = float("inf")
                best_dPtRel = float("inf")

                # Loop over unmatched PF candidates
                for i_pf, pf in enumerate(pf_candidates):

                    dR = deltaR(geta, gphi, pf["eta"], pf["phi"])
                    if dR >= 0.1:
                        continue

                    dPtRel = abs(gpt - pf["pt"]) / gpt
                    if dPtRel >= 0.3:
                        continue

                    # Take PF Cand closest to the gen particle. If tied, take pf cand with closest pt
                    if dR < best_dR or (dR == best_dR and dPtRel < best_dPtRel):
                        best_match = i_pf
                        best_dR = dR
                        best_dPtRel = dPtRel

                if best_match is not None:
                    n_matched += 1
                    matched_gen_pts.append(gpt)

                    # Remove PF candidate once it's been matched
                    pf_candidates.pop(best_match)

            # Initialize tracking efficiency denominator: number of all gen muons passing object and event level selections
            n_gen = len(track_eff_denominator_muons)
            n_gen_total += n_gen

            # Initialize tracking efficiency numerator: number of all gen muons passing object and event level selections matched to a PF track
            n_matched_total += n_matched

            # Event loop complete!
            iEv += 1

        # Write efficiencies out per file
        track_efficiency = n_matched_total / n_gen_total if n_gen_total > 0 else 0
        print(f"Gen muons (OSSF): {n_gen_total}")
        print(f"Matched PF muons: {n_matched_total}")
        print(f"Track efficiency: {track_efficiency:.4f}")
        tf.Close()

    print(f"Finished processing {inF} files with {iEv} events.")
    print(f"Total OSSF pairs found: {len(masses)}")

    # Plotting
    cmap = plt.get_cmap("tab10")
    hep.style.use("CMS")

    # Dilepton mass
    plt.figure(figsize=(8, 8))
    hep.cms.label(data = True, label="Preliminary")
    plt.hist(masses, bins=50, range=(3, 3.3), histtype='step', color=cmap(0), linewidth=2)
    plt.xlabel("Dilepton Mass [GeV]")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig("dilepton_mass.png")
    plt.close()

    # pT
    plt.figure(figsize=(8, 8))
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
    plt.figure(figsize=(8, 8))
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
    plt.figure(figsize=(8, 8))
    hep.cms.label(data = True, label="Preliminary")
    plt.hist(lead_phis, bins=20, range=(-np.pi, np.pi), histtype='step', label="Leading", color=cmap(0), linewidth=2)
    plt.hist(sub_phis, bins=20, range=(-np.pi, np.pi), histtype='step', label="Subleading", color=cmap(2), linewidth=2)
    plt.xlabel("Muon φ")
    plt.ylabel("Counts")
    plt.legend()
    # plt.tight_layout()
    plt.savefig("muon_phi.png")
    plt.close()

    # Track efficiency
    bins = np.linspace(0, 50, 21)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    denom, _ = np.histogram(gen_pts, bins=bins)
    num, _ = np.histogram(matched_gen_pts, bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = np.true_divide(num, denom)
        efficiency[denom == 0] = np.nan
        cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(8, 8))
    hep.cms.label(data = True, label="Preliminary")
    plt.plot(bin_centers, efficiency, marker='o', linestyle='-', color='darkgreen')
    err = np.sqrt(efficiency * (1 - efficiency) / denom)
    err[denom == 0] = 0
    plt.errorbar(bin_centers, efficiency, yerr=err, fmt='o', color='blue')
    plt.xlabel("Gen Muon $p_T$ [GeV]")
    plt.ylabel("Tracking Efficiency")
    plt.savefig("tracking_efficiency_vs_pt.pdf")
    plt.close()

    print("Saved all plots.")

if __name__ == "__main__":
    main()
