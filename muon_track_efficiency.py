import ROOT
import os
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
from array import array

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
    track_eff_den_pts, track_eff_num_pts = [], []
    masses = []

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
        iEv = 0
        
        # Start looping through the events in a given file
        for ev in events:

            # For quick testing
            iEv += 1
            # if(iEv % 100 == 0):
            #     break

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
            status = ev.GenPart_status
            masses_branch = ev.GenPart_mass

            # Create gen muon collection
            gen_muons = []
            for i in range(nGenPart):
                # Apply object-level selections
                if abs(pdgIds[i]) == 13 and abs(etas[i]) < 2.4 and status[i] == 1:
                    gen_muons.append({
                        "pt": pts[i],
                        "eta": etas[i],
                        "phi": phis[i],
                        "mass": masses_branch[i],
                        "pdgId": pdgIds[i],
                        "index": i
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
            gen_muons_for_eff = set()
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
                    if inv_mass < 60 or inv_mass > 120:
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
                    gen_muons_for_eff.add(mu1["index"])
                    gen_muons_for_eff.add(mu2["index"])

            # Build gen-level muon kinematic lists
            track_eff_denominator_muons = gen_muons_for_eff
            track_eff_denominator_pts = [pts[i] for i in track_eff_denominator_muons]
            track_eff_denominator_etas = [etas[i] for i in track_eff_denominator_muons]
            track_eff_denominator_phis = [phis[i] for i in track_eff_denominator_muons]

            # Store tracking efficiency denominator
            track_eff_den_pts.extend(track_eff_denominator_pts)

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
                    track_eff_num_pts.append(gpt)

                    # Remove PF candidate once it's been matched
                    pf_candidates.pop(best_match)

            # Initialize tracking efficiency denominator: number of all gen muons passing object and event level selections
            n_gen = len(track_eff_denominator_muons)
            n_gen_total += n_gen

            # Initialize tracking efficiency numerator: number of all gen muons passing object and event level selections matched to a PF track
            n_matched_total += n_matched

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
    hep.cms.label(data = True, label="Preliminary", year = 2018)
    plt.hist(masses, bins=50, range=(0, 150), histtype='step', color=cmap(0), linewidth=2)
    plt.xlabel(r"$m_{\mu\mu}\,\mathrm{[GeV]}$")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig("dilepton_mass.png")
    plt.close()

    # pT
    plt.figure(figsize=(8, 8))
    hep.cms.label(data = True, label="Preliminary", year = 2018)
    plt.hist(lead_pts, bins=20, range=(0, 200), histtype='step', label="Leading", color=cmap(0), linewidth=2)
    plt.hist(sub_pts, bins=20, range=(0, 200), histtype='step', label="Subleading", color=cmap(2), linewidth=2)
    plt.xlabel(r"$p_T^{\mu}\,\mathrm{[GeV]}$")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig("muon_pt.png")
    plt.close()

    # Eta
    plt.figure(figsize=(8, 8))
    hep.cms.label(data = True, label="Preliminary", year = 2018)
    plt.hist(lead_etas, bins=20, range=(-3, 3), histtype='step', label="Leading", color=cmap(0), linewidth=2)
    plt.hist(sub_etas, bins=20, range=(-3, 3), histtype='step', label="Subleading", color=cmap(2), linewidth=2)
    plt.xlabel(r"$\eta^{\mu}$")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig("muon_eta.png")
    plt.close()

    # Phi
    plt.figure(figsize=(8, 8))
    hep.cms.label(data = True, label="Preliminary", year = 2018)
    plt.hist(lead_phis, bins=20, range=(-np.pi, np.pi), histtype='step', label="Leading", color=cmap(0), linewidth=2)
    plt.hist(sub_phis, bins=20, range=(-np.pi, np.pi), histtype='step', label="Subleading", color=cmap(2), linewidth=2)
    plt.xlabel(r"$\phi^{\mu}$")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig("muon_phi.png")
    plt.close()

    # Track efficiency
    bins = np.linspace(0, 325, 25)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    denom, _ = np.histogram(track_eff_den_pts, bins=bins)
    num, _ = np.histogram(track_eff_num_pts, bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = np.true_divide(num, denom)
        efficiency[denom == 0] = np.nan
        cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(8, 8))
    hep.cms.label(data = True, label="Preliminary", year = 2018)
    plt.plot(bin_centers, efficiency, marker='o', linestyle='-', color='darkgreen')
    err = np.sqrt(efficiency * (1 - efficiency) / denom)
    err[denom == 0] = 0
    plt.errorbar(bin_centers, efficiency, yerr=err, fmt='o', color='blue')
    plt.xlabel(r"$p_T^{\mu}\,\mathrm{[GeV]}$")
    plt.ylabel("Tracking Efficiency")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig("tracking_efficiency_vs_pt.png")
    plt.close()

    print("Saved all plots.")

if __name__ == "__main__":
    main()
