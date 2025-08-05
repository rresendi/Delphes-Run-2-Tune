import ROOT
import os

# Open file
input_dir = "/eos/user/g/gdecastr/sampleFactoryOutput/SampleFactory/ZPrimeFragment/chain_BPH-RunIISummer20UL18GEN-00263-BPH-RunIISummer20UL18MiniAODv2-00292/20250805_050441/NanoAOD"
output_file = "passing_leptons.txt"
lepton_type = "Muon"  # or "Muon" if you change the cut function

# Loopthrough files in dir
input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".root")]

# Define muon
def passes_lepton_cuts(ev, lepton, leptonIndex):
    if lepton == "Muon":
        return (
            abs(ev.GenPart_pdgId[leptonIndex]) == 11
            and abs(ev.GenPart_eta[leptonIndex]) < 2.4
        )
    return False

# Counts
total_events = 0
passing_leptons = 0

# Open file
with open(output_file, "w") as out:
    out.write("# Event  LeptonIdx  eta     phi pt\n")

    # Loop through files in dir
    for file_path in input_files:
        file = ROOT.TFile.Open(file_path)

        if not file or file.IsZombie():
            print(f"Could not open {file_path}. Skipping.")
            continue

        events = file.Get("Events")
        if not events:
            print(f"No 'Events' tree found in {file_path}. Skipping.")
            continue

        n_events = events.GetEntries()
        total_events += n_events

        for i in range(n_events):
            events.GetEntry(i)
            nGenPart = events.nGenPart
            for idx in range(nGenPart):
                if passes_lepton_cuts(events, lepton_type, idx):
                    eta = events.GenPart_eta[idx]
                    phi = events.GenPart_phi[idx]
                    pt = events.GenPart_pt[idx]
                    passing_leptons += 1
                    out.write(f"{i:<7d} {idx:<10d} {eta:6.3f} {phi:6.3f} {pt:6.3f}\n")

        file.Close()

# Prints
print(f"\nTotal number of events: {total_events}")
print(f"Total number of {lepton_type}s passing cuts: {passing_leptons}")
print(f"Results written to: {output_file}")
