import ROOT
import os
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
from array import array
import sys

def main():
    input_dir = sys.argv[1]
    inputFiles = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".root")]

    inF = 0
    iEv = 0
    nF = len(inputFiles)

    # Object system argument
    obj = sys.argv[2] # electron or muon

    # Sample system argument
    samp = sys.argv[3] # dy, jpsi, or zprime

    # Output lists
    lead_pts, sub_pts, gen_pts = [], [], []
    lead_etas, sub_etas, gen_etas = [], [], []
    lead_phis, sub_phis, gen_phis = [], [], []
    masses = []
    calo_eff_den_pts, calo_eff_den_etas, calo_eff_num_pts, calo_eff_num_etas = [], [], [], []
    track_eff_den_etas, track_eff_den_pts, track_eff_num_pts, track_eff_num_etas = [], [], [], []

    # Initialize counts
    n_matched = 0

    # Create 2D histograms for numerator and denominator of tracker efficiency
    if obj == "muon":

        # Set up 2D histograms with binning used in current nominal Delphes card
        track_eta_edges = array('d', [0, 0.9, 1.2, 2.1, 2.4])
        track_pt_edges = array('d', [0, 0.1, 1.0, 5.0, 10.0, 100, 1000, 2000])

        calo_eta_edges = array('d', [0, 0.9, 1.2, 2.1, 2.4])
        calo_pt_edges = array('d', [0, 0.1, 1.0, 5.0, 10.0, 100, 1000, 2000])

        hTrackEffNum = ROOT.TH2F("hTrackEffNum", "Tracker Efficiency Numerator;|#eta|;p_{T} [GeV]",
                                len(track_eta_edges)-1, track_eta_edges,
                                len(track_pt_edges)-1, track_pt_edges)
        hTrackEffDen = ROOT.TH2F("hTrackEffDen", "Tracker Efficiency Denominator;|#eta|;p_{T} [GeV]",
                                len(track_eta_edges)-1, track_eta_edges,
                                len(track_pt_edges)-1, track_pt_edges)

        # Create 2D histograms for numerator and denominator of calorimeter (chamber) efficiency
        hCaloEffNum = ROOT.TH2F("hCaloEffNum", "Calorimeter Efficiency Numerator;|#eta|;p_{T} [GeV]",
                            len(calo_eta_edges)-1, calo_eta_edges,
                            len(calo_pt_edges)-1, calo_pt_edges)
        hCaloEffDen = ROOT.TH2F("hCaloEffDen", "Calorimeter Efficiency Denominator;|#eta|;p_{T} [GeV]",
                            len(calo_eta_edges)-1, calo_eta_edges,
                            len(calo_pt_edges)-1, calo_pt_edges)

    elif obj == "electron":

        # Set up 2D histograms with binning used in current nominal Delphes card
        track_eta_edges = array('d', [0, 1, 1.444, 1.566, 2.0, 3.0])
        track_pt_edges = array('d', [0, 0.1, 1.0, 5.0, 10.0, 100, 2000])

        calo_eta_edges = array('d', [0, 1, 1.444, 1.566, 2.0, 3.0])
        calo_pt_edges = array('d', [0, 0.1, 1.0, 5.0, 10.0, 100, 2000])

        hTrackEffNum = ROOT.TH2F("hTrackEffNum", "Tracker Efficiency Numerator;|#eta|;p_{T} [GeV]",
                                len(track_eta_edges)-1, track_eta_edges,
                                len(track_pt_edges)-1, track_pt_edges)
        hTrackEffDen = ROOT.TH2F("hTrackEffDen", "Tracker Efficiency Denominator;|#eta|;p_{T} [GeV]",
                                len(track_eta_edges)-1, track_eta_edges,
                                len(track_pt_edges)-1, track_pt_edges)

        # Create 2D histograms for numerator and denominator of calorimeter (chamber) efficiency
        hCaloEffNum = ROOT.TH2F("hCaloEffNum", "Calorimeter Efficiency Numerator;|#eta|;p_{T} [GeV]",
                            len(calo_eta_edges)-1, calo_eta_edges,
                            len(calo_pt_edges)-1, calo_pt_edges)
        hCaloEffDen = ROOT.TH2F("hCaloEffDen", "Calorimeter Efficiency Denominator;|#eta|;p_{T} [GeV]",
                            len(calo_eta_edges)-1, calo_eta_edges,
                            len(calo_pt_edges)-1, calo_pt_edges)


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
        try:
            tf = ROOT.TFile.Open(iFile, "READ")
        except OSError as e:
            print(f"Error opening file {iFile}: {e}. Skipping.")
            continue

        if not tf or tf.IsZombie():
            print(f"File {iFile} is unreadable or corrupted. Skipping.")
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
            # if(iEv % 1000 == 0):
            #     break

            # Initialize tracking efficiency denominator
            track_eff_denominator_objs = set()

            # Initialize n_matched for each event
            n_matched = 0

            # Reset 2D histogram numerators and denominators per event
            track_hist_den_etas, track_hist_den_pts, track_hist_num_pts, track_hist_num_etas = [], [], [], []
            calo_hist_den_pts, calo_hist_den_etas, calo_hist_num_pts, calo_hist_num_etas = [], [], [], []

            # Extract relevant branches
            nGenPart = ev.nGenPart
            pdgIds = ev.GenPart_pdgId
            pts = ev.GenPart_pt
            etas = ev.GenPart_eta
            phis = ev.GenPart_phi
            status = ev.GenPart_status
            masses_branch = ev.GenPart_mass

            # Create gen muon collection
            gen_objs = []
            if obj == "muon":
                for i in range(nGenPart):
                    # Apply object-level selections
                    if abs(pdgIds[i]) == 13 and abs(etas[i]) < 2.4 and status[i] == 1:
                        gen_objs.append({
                            "pt": pts[i],
                            "eta": etas[i],
                            "phi": phis[i],
                            "mass": masses_branch[i],
                            "pdgId": pdgIds[i],
                            "index": i
                        })

            elif obj == "electron":
                for i in range(nGenPart):
                    # Apply object-level selections
                    if abs(pdgIds[i]) == 11 and abs(etas[i]) < 2.5 and status[i] == 1:
                        gen_objs.append({
                            "pt": pts[i],
                            "eta": etas[i],
                            "phi": phis[i],
                            "mass": masses_branch[i],
                            "pdgId": pdgIds[i],
                            "index": i
                        })

            # Sort by pt
            gen_objs = sorted(gen_objs, key=lambda x: x["pt"], reverse=True)

            # Create gen muon list once per event
            for gm in gen_objs:
                gen_pts.append(gm["pt"])
                gen_etas.append(gm["eta"])
                gen_phis.append(gm["phi"])

            if len(gen_objs) < 2:
                continue

            # Initialize PF Candidate list
            pf_candidates = []

            # Loop through subleading leptons to create OSSF pairs
            gen_objs_for_eff = set()
            for i, obj1 in enumerate(gen_objs):
                # Do the OSSF check
                for obj2 in gen_objs[i+1:]:
                    if obj1["pdgId"] * obj2["pdgId"] >= 0:
                        continue

                    # Calculate dilepton invariant mass only using OSSF pairs
                    p4_1 = ROOT.TLorentzVector()
                    p4_1.SetPtEtaPhiM(obj1["pt"], obj1["eta"], obj1["phi"], obj1["mass"])
                    p4_2 = ROOT.TLorentzVector()
                    p4_2.SetPtEtaPhiM(obj2["pt"], obj2["eta"], obj2["phi"], obj2["mass"])

                    # Event level selection that only takes pairs on the appropriate mass peak
                    inv_mass = (p4_1 + p4_2).M()

                    if samp == "jpsi":
                        if inv_mass < 2.9 or inv_mass > 3.3:
                            continue

                    elif samp == "dy":
                        if inv_mass < 60 or inv_mass > 120:
                            continue
                    
                    elif samp == "zprime":
                        if inv_mass < 2300 or inv_mass > 2700:
                            continue

                    # Appending to lists for plotting
                    masses.append(inv_mass)
                    lead_pts.append(obj1["pt"])
                    lead_etas.append(obj1["eta"])
                    lead_phis.append(obj1["phi"])
                    sub_pts.append(obj2["pt"])
                    sub_etas.append(obj2["eta"])
                    sub_phis.append(obj2["phi"])

                    # Adding muons to the OSSF gen-muons set
                    gen_objs_for_eff.add(obj1["index"])
                    gen_objs_for_eff.add(obj2["index"])

            # Build gen-level muon kinematic lists
            track_eff_denominator_objs = gen_objs_for_eff
            track_eff_denominator_pts = [pts[i] for i in track_eff_denominator_objs]
            track_eff_denominator_etas = [etas[i] for i in track_eff_denominator_objs]
            track_eff_denominator_phis = [phis[i] for i in track_eff_denominator_objs]

            # Store tracking efficiency denominator
            track_eff_den_pts.extend(track_eff_denominator_pts)
            track_hist_den_pts.extend(track_eff_denominator_pts)
            track_hist_den_etas.extend(track_eff_denominator_etas)

            # Initialize a list of PF candidates
            if obj == "muon":
                for i in range(ev.nPFCands):
                    # if abs(ev.PFCands_pdgId[i]) == 13:
                    pf_candidates.append({
                    "index": i,
                    "pt": ev.PFCands_pt[i],
                    "eta": ev.PFCands_eta[i],
                    "phi": ev.PFCands_phi[i]
                    })

            elif obj == "electron":
                for i in range(ev.nPFCands):
                    # if abs(ev.PFCands_pdgId[i]) == 11:
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
                    if obj == "muon":
                        if dPtRel >= 0.3:
                            continue

                    # Take PF Cand closest to the gen particle. If tied, take pf cand with closest pt
                    if dR < best_dR or (dR == best_dR and dPtRel < best_dPtRel):
                        best_match = i_pf
                        best_dR = dR
                        best_dPtRel = dPtRel

                if best_match is not None:
                    n_matched += 1
                    
                    # Append pT value to the numerator
                    track_eff_num_pts.append(gpt)
                    track_eff_num_etas.append(geta)
                    track_hist_num_pts.append(gpt)
                    track_hist_num_etas.append(geta)

                    # Create denominator collection for chamber efficiency
                    pf_matched = pf_candidates[best_match]
                    calo_eff_den_pts.append(pf_matched["pt"])
                    calo_hist_den_pts.append(pf_matched["pt"])
                    calo_eff_den_etas.append(pf_matched["eta"])
                    calo_hist_den_etas.append(pf_matched["eta"])

                    # Match PF tracks to fully reconstructed muons
                    matched_to_reco = False
                    if obj == "muon":
                        for i_mu in range(ev.nMuon):
                            if abs(ev.Muon_pdgId[i_mu]) != 13:
                                continue
                            
                            reco_eta = ev.Muon_eta[i_mu]
                            reco_phi = ev.Muon_phi[i_mu]
                            reco_pt = ev.Muon_pt[i_mu]

                            dR = deltaR(pf_matched["eta"], pf_matched["phi"], reco_eta, reco_phi)
                            if dR >= 0.1:
                                continue
                            
                            dPtRel = abs(pf_matched["pt"] - reco_pt) / pf_matched["pt"]
                            if dPtRel >= 0.3:
                                continue

                            matched_to_reco = True
                            break

                    elif obj == "electron":
                        for i_ele in range(ev.nElectron):
                            # if abs(ev.Electron_pdgId[i_ele]) != 11:
                            #     continue
                            
                            reco_eta = ev.Electron_eta[i_ele]
                            reco_phi = ev.Electron_phi[i_ele]
                            reco_pt = ev.Electron_pt[i_ele]

                            dR = deltaR(pf_matched["eta"], pf_matched["phi"], reco_eta, reco_phi)
                            if dR >= 0.1:
                                continue
                            
                            dPtRel = abs(pf_matched["pt"] - reco_pt) / pf_matched["pt"]
                            if dPtRel >= 0.3:
                                continue

                            matched_to_reco = True
                            break
                    
                    if matched_to_reco:
                        calo_eff_num_pts.append(pf_matched["pt"])
                        calo_eff_num_etas.append(pf_matched["eta"])
                        calo_hist_num_pts.append(pf_matched["pt"])
                        calo_hist_num_etas.append(pf_matched["eta"])

                    # Remove PF candidate once it's been matched
                    pf_candidates.pop(best_match)

            # Fill tracking efficiency denominator: number of all gen muons passing object and event level selections
            n_gen = len(track_eff_denominator_objs)
            n_gen_total += n_gen

            # Fill tracking efficiency numerator: number of all gen muons passing object and event level selections matched to a PF track
            n_matched_total += n_matched

            # Fill 2D track histogram denominator
            for eta, pt in zip(track_hist_den_etas, track_hist_den_pts):
                hTrackEffDen.Fill(abs(eta), pt)

            # Fill 2D track histogram numerator
            for eta, pt in zip(track_hist_num_etas, track_hist_num_pts):
                hTrackEffNum.Fill(abs(eta), pt)

            # Fill 2D calo efficiency denominator
            for eta, pt in zip(calo_hist_den_etas, calo_hist_den_pts):
                hCaloEffDen.Fill(abs(eta), pt)

            # Fill 2D calo efficiency numerator
            for eta, pt in zip(calo_hist_num_etas, calo_hist_num_pts):
                hCaloEffNum.Fill(abs(eta), pt)

        # Write efficiencies out per file
        track_efficiency = n_matched_total / n_gen_total if n_gen_total > 0 else 0
        print(f"Gen muons (OSSF): {n_gen_total}")
        print(f"Matched PF {obj}: {n_matched_total}")
        print(f"Track efficiency: {track_efficiency:.4f}")
        tf.Close()

    print(f"Finished processing {inF} files with {iEv} events.")
    print(f"Total OSSF pairs found: {len(masses)}")

    # Write out 2D histograms to ROOT file
    if obj == "muon":
        fout = ROOT.TFile("muon_2D_efficiency_histograms.root", "RECREATE")

    elif obj == "electron":
        fout = ROOT.TFile("electron_2D_efficiency_histograms.root", "RECREATE")
    hTrackEffNum.Write()
    hTrackEffDen.Write()
    hCaloEffNum.Write()
    hCaloEffDen.Write()
    fout.Close()

    # Plotting
    cmap = plt.get_cmap("tab10")
    hep.style.use("CMS")

    # Dilepton mass
    plt.figure(figsize=(8, 8))
    hep.cms.label(data = True, label="Preliminary", year = 2018)
    if samp == "dy":
        plt.hist(masses, bins=50, range=(0, 150), histtype='step', color=cmap(0), linewidth=2)
    elif samp == "jpsi":
        plt.hist(masses, bins=21, range=(2.7, 3.4), histtype='step', color=cmap(0), linewidth=2)
    elif samp == "zprime":
            plt.hist(masses, bins=21, range=(2250, 2750), histtype='step', color=cmap(0), linewidth=2)
    if obj == "muon":
        plt.xlabel(r"$m_{\mu\mu}\,\mathrm{[GeV]}$")
    elif obj == "electron":
        plt.xlabel(r"$m_{ee}\,\mathrm{[GeV]}$")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig(f"{samp}_{obj}_dilepton_mass.png")
    plt.close()

    # pT
    plt.figure(figsize=(8, 8))
    hep.cms.label(data = True, label="Preliminary", year = 2018)
    if samp == "dy":
        plt.hist(lead_pts, bins=20, range=(0, 200), histtype='step', label="Leading", color=cmap(0), linewidth=2)
        plt.hist(sub_pts, bins=20, range=(0, 200), histtype='step', label="Subleading", color=cmap(2), linewidth=2)
    elif samp == "jpsi":
        plt.hist(lead_pts, bins=10, range=(0, 10), histtype='step', label="Leading", color=cmap(0), linewidth=2)
        plt.hist(sub_pts, bins=10, range=(0, 10), histtype='step', label="Subleading", color=cmap(2), linewidth=2)
    elif samp == "zprime":
        plt.hist(lead_pts, bins=10, range=(950, 1500), histtype='step', label="Leading", color=cmap(0), linewidth=2)
        plt.hist(sub_pts, bins=10, range=(950, 1500), histtype='step', label="Subleading", color=cmap(2), linewidth=2)
    if obj == "muon":
        plt.xlabel(r"$p_T^{\mu}\,\mathrm{[GeV]}$")
    elif obj == "electron":
        plt.xlabel(r"$p_T^{e}\,\mathrm{[GeV]}$")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{samp}_{obj}_pt.png")
    plt.close()

    # Eta
    plt.figure(figsize=(8, 8))
    hep.cms.label(data = True, label="Preliminary", year = 2018)
    if samp in ["dy", "zprime"]:
        plt.hist(lead_etas, bins=20, range=(-3, 3), histtype='step', label="Leading", color=cmap(0), linewidth=2)
        plt.hist(sub_etas, bins=20, range=(-3, 3), histtype='step', label="Subleading", color=cmap(2), linewidth=2)
    elif samp == "jpsi":
        plt.hist(lead_etas, bins=40, range=(-3, 3), histtype='step', label="Leading", color=cmap(0), linewidth=2)
        plt.hist(sub_etas, bins=40, range=(-3, 3), histtype='step', label="Subleading", color=cmap(2), linewidth=2)
    if obj == "muon":
        plt.xlabel(r"$\eta^{\mu}$")
    elif obj == "electron":
        plt.xlabel(r"$\eta^{e}$")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{samp}_{obj}_eta.png")
    plt.close()

    # Phi
    plt.figure(figsize=(8, 8))
    hep.cms.label(data = True, label="Preliminary", year = 2018)
    if samp in ["dy", "zprime"]:
        plt.hist(lead_phis, bins=20, range=(-np.pi, np.pi), histtype='step', label="Leading", color=cmap(0), linewidth=2)
        plt.hist(sub_phis, bins=20, range=(-np.pi, np.pi), histtype='step', label="Subleading", color=cmap(2), linewidth=2)
    elif samp == "jpsi":
        plt.hist(lead_phis, bins=40, range=(-np.pi, np.pi), histtype='step', label="Leading", color=cmap(0), linewidth=2)
        plt.hist(sub_phis, bins=40, range=(-np.pi, np.pi), histtype='step', label="Subleading", color=cmap(2), linewidth=2)
    if obj == "muon":
        plt.xlabel(r"$\phi^{\mu}$")
    elif obj == "electron":
        plt.xlabel(r"$\phi^{e}$")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{samp}_{obj}_phi.png")
    plt.close()

    # Track efficiency
    if samp == "dy":
        bins = np.linspace(0, 325, 25)
    elif samp == "jpsi":
        bins = np.linspace(0, 11, 11)
    elif samp == "zprime":
        bins = np.linspace(950, 2000, 25)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    denom, _ = np.histogram(track_eff_den_pts, bins=bins)
    num, _ = np.histogram(track_eff_num_pts, bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        track_efficiency = np.true_divide(num, denom)
        track_efficiency[denom == 0] = np.nan
        cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(8, 8))
    hep.cms.label(data = True, label="Preliminary", year = 2018)
    plt.plot(bin_centers, track_efficiency, marker='o', linestyle='-', color='darkgreen')
    err = np.sqrt(track_efficiency * (1 - track_efficiency) / denom)
    err[denom == 0] = 0
    plt.errorbar(bin_centers, track_efficiency, yerr=err, fmt='o', color='blue')
    if obj == "muon":
        plt.xlabel(r"$p_T^{\mu}\,\mathrm{[GeV]}$")
    elif obj == "electron":
        plt.xlabel(r"$p_T^{e}\,\mathrm{[GeV]}$")
    plt.ylabel("Tracking Efficiency")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f"{samp}_{obj}_tracking_efficiency_vs_pt.png")
    plt.close()

    # Chamber efficiency
    if samp == "dy":
        bins = np.linspace(0, 325, 25)
    elif samp == "jpsi":
        bins = np.linspace(0, 11, 11)
    elif samp == "zprime":
        bins = np.linspace(950, 2000, 25)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    denom, _ = np.histogram(calo_eff_den_pts, bins=bins)
    num, _ = np.histogram(calo_eff_num_pts, bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        calo_efficiency = np.true_divide(num, denom)
        calo_efficiency[denom == 0] = np.nan
        cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(8, 8))
    hep.cms.label(data = True, label="Preliminary", year = 2018)
    plt.plot(bin_centers, calo_efficiency, marker='o', linestyle='-', color='darkgreen')
    err = np.sqrt(calo_efficiency * (1 - calo_efficiency) / denom)
    err[denom == 0] = 0
    plt.errorbar(bin_centers, calo_efficiency, yerr=err, fmt='o', color='blue')
    plt.ylim(0, 1.1)
    if obj == "muon":
        plt.xlabel(r"$p_T^{\mu}\,\mathrm{[GeV]}$")
        plt.ylabel("Muon Chamber Efficiency")
        plt.tight_layout()
        plt.savefig(f"{samp}_muon_chamber_efficiency_vs_pt.png")
    elif obj == "electron":
        plt.xlabel(r"$p_T^{e}\,\mathrm{[GeV]}$")
        plt.ylabel("ECAL Efficiency")
        plt.tight_layout()
        plt.savefig(f"{samp}_ECAL_efficiency_vs_pt.png")
    plt.close()

    print("Saved all plots.")

if __name__ == "__main__":
    main()
