import ROOT
import os

# Path to the directory containing ROOT files
input_dir = "/eos/user/g/gdecastr/sampleFactoryOutput/SampleFactory/JPsi_DiMu_Pythia8/chain_SMP-RunIISummer20UL18GEN-00050-SMP-RunIISummer20UL18NanoAODv9-00368/20250726_062214/"

# Get all .root files in the directory
input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".root")]

# Total event counter
total_events = 0

# Loop over each ROOT file
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

    # print(f"{file_path}: {n_events} events")
    file.Close()

print(f"\nTotal number of events across all files: {total_events}")
