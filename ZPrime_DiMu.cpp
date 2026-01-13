#include <iostream>
#include <cmath> // Prefer <cmath> over <math.h> in C++
#include <string>
#include <cstdio> // Prefer <cstdio> over <stdio.h> in C++
#include <cstdlib> // Prefer <cstdlib> over <stdlib.h> in C++

// Boost headers
#include <boost/math/tools/roots.hpp>
#include <boost/bind/bind.hpp> // Updated include
using namespace boost::placeholders; // Use the recommended namespace for placeholders

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"

using namespace Pythia8;

// Apply Z' -> mu+ mu- filter similar to CMSSW fragment
static bool passZPrimeMuMu(const Pythia8::Event& ev) {
    const int zprimeId = 32;      // Z'
    const double minPt = 1000.0;  // GeV
    const double maxAbsEta = 2.5;

    for (int i = 0; i < ev.size(); ++i) {
        if (ev[i].id() != zprimeId) continue; // require Z' mother
        const std::vector<int>& dlist = ev[i].daughterList();
        if (dlist.size() != 2) continue; // CMSSW filter expects exactly two specified daughters
        int dA = dlist[0];
        int dB = dlist[1];
        if (dA <= 0 || dB <= 0 || dA >= ev.size() || dB >= ev.size()) continue;

        int idA = ev[dA].id();
        int idB = ev[dB].id();
        bool isMuPair = ( (idA == 13 && idB == -13) || (idA == -13 && idB == 13) );
        if (!isMuPair) continue;

        double ptA  = ev[dA].pT();
        double ptB  = ev[dB].pT();
        double etaA = ev[dA].eta();
        double etaB = ev[dB].eta();
        if (ptA >= minPt && ptB >= minPt && std::abs(etaA) <= maxAbsEta && std::abs(etaB) <= maxAbsEta) {
            return true;
        }
    }
    return false;
}

int main(int argc, char *argv[]) {

    // Make sure we have the correct arguments: output file, random seed, number of events
    if (argc != 4) {
        std::cout << "Usage: " << argv[0]
                << " <outputHepMCfile> <randomSeed> <nEvents>\n";
        return 1;
    }

    // Parse arguments
    std::string outFile  = argv[1];
    std::string seedStr  = argv[2];
    int nEvent           = std::stoi(argv[3]);

    // Initialize Pythia
    Pythia pythia;

    // Basic setup for 13 TeV collisions
    pythia.readString("Beams:eCM = 13000.");

    // Z' production and decays (physics only — no tune changes)
    pythia.readString("NewGaugeBoson:ffbar2gmZZprime = on");
    pythia.readString("Zprime:gmZmode = 3");

    // Z' properties
    pythia.readString("32:m0 = 5000.0");

    // Force Z' → ℓℓ only (allow both e+e- and μ+μ-, filtering to μμ below)
    pythia.readString("32:onMode = off");
    pythia.readString("32:onIfMatch = 11 -11");
    pythia.readString("32:onIfMatch = 13 -13");

    // Random seed settings
    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = " + seedStr);

    // To mimic CMSSW config
    pythia.readString("Tune:preferLHAPDF = 2"); 
    pythia.readString("Main:timesAllowErrors = 10000");
    pythia.readString("Check:epTolErr = 0.01");
    pythia.readString("Beams:setProductionScalesFromLHEF = off");
    pythia.readString("SLHA:minMassSM = 1000.");
    pythia.readString("ParticleDecays:limitTau0 = on");
    pythia.readString("ParticleDecays:tau0Max = 10");
    pythia.readString("ParticleDecays:allowPhotonRadiation = on");
    pythia.readString("Tune:pp 14");
    pythia.readString("Tune:ee 7");
    pythia.readString("MultipartonInteractions:ecmPow=0.03344");
    pythia.readString("MultipartonInteractions:bProfile=2");
    pythia.readString("MultipartonInteractions:pT0Ref=1.41");
    pythia.readString("MultipartonInteractions:coreRadius=0.7634");
    pythia.readString("MultipartonInteractions:coreFraction=0.63");
    pythia.readString("ColourReconnection:range=5.176");
    pythia.readString("SigmaTotal:zeroAXB=off");
    pythia.readString("SpaceShower:alphaSorder=2");
    pythia.readString("SpaceShower:alphaSvalue=0.118");
    pythia.readString("SigmaProcess:alphaSvalue=0.118");
    pythia.readString("SigmaProcess:alphaSorder=2");
    pythia.readString("MultipartonInteractions:alphaSvalue=0.118");
    pythia.readString("MultipartonInteractions:alphaSorder=2");
    pythia.readString("TimeShower:alphaSorder=2");
    pythia.readString("TimeShower:alphaSvalue=0.118");
    pythia.readString("SigmaTotal:mode = 0");
    pythia.readString("SigmaTotal:sigmaEl = 21.89");
    pythia.readString("SigmaTotal:sigmaTot = 100.309");
    pythia.readString("PDF:pSet=LHAPDF6:NNPDF31_nnlo_as_0118"); // Might be the issue with the docker image

    pythia.readString("UncertaintyBands:doVariations = on");
    pythia.readString("UncertaintyBands:List = {\
        isrRedHi isr:muRfac=0.707,fsrRedHi fsr:muRfac=0.707,isrRedLo isr:muRfac=1.414,fsrRedLo fsr:muRfac=1.414,\
        isrDefHi isr:muRfac=0.5,fsrDefHi fsr:muRfac=0.5,isrDefLo isr:muRfac=2.0,fsrDefLo fsr:muRfac=2.0,\
        isrConHi isr:muRfac=0.25,fsrConHi fsr:muRfac=0.25,isrConLo isr:muRfac=4.0,fsrConLo fsr:muRfac=4.0,\
        fsr_G2GG_muR_dn fsr:G2GG:muRfac=0.5,\
        fsr_G2GG_muR_up fsr:G2GG:muRfac=2.0,\
        fsr_G2QQ_muR_dn fsr:G2QQ:muRfac=0.5,\
        fsr_G2QQ_muR_up fsr:G2QQ:muRfac=2.0,\
        fsr_Q2QG_muR_dn fsr:Q2QG:muRfac=0.5,\
        fsr_Q2QG_muR_up fsr:Q2QG:muRfac=2.0,\
        fsr_X2XG_muR_dn fsr:X2XG:muRfac=0.5,\
        fsr_X2XG_muR_up fsr:X2XG:muRfac=2.0,\
        fsr_G2GG_cNS_dn fsr:G2GG:cNS=-2.0,\
        fsr_G2GG_cNS_up fsr:G2GG:cNS=2.0,\
        fsr_G2QQ_cNS_dn fsr:G2QQ:cNS=-2.0,\
        fsr_G2QQ_cNS_up fsr:G2QQ:cNS=2.0,\
        fsr_Q2QG_cNS_dn fsr:Q2QG:cNS=-2.0,\
        fsr_Q2QG_cNS_up fsr:Q2QG:cNS=2.0,\
        fsr_X2XG_cNS_dn fsr:X2XG:cNS=-2.0,\
        fsr_X2XG_cNS_up fsr:X2XG:cNS=2.0,\
        isr_G2GG_muR_dn isr:G2GG:muRfac=0.5,\
        isr_G2GG_muR_up isr:G2GG:muRfac=2.0,\
        isr_G2QQ_muR_dn isr:G2QQ:muRfac=0.5,\
        isr_G2QQ_muR_up isr:G2QQ:muRfac=2.0,\
        isr_Q2QG_muR_dn isr:Q2QG:muRfac=0.5,\
        isr_Q2QG_muR_up isr:Q2QG:muRfac=2.0,\
        isr_X2XG_muR_dn isr:X2XG:muRfac=0.5,\
        isr_X2XG_muR_up isr:X2XG:muRfac=2.0,\
        isr_G2GG_cNS_dn isr:G2GG:cNS=-2.0,\
        isr_G2GG_cNS_up isr:G2GG:cNS=2.0,\
        isr_G2QQ_cNS_dn isr:G2QQ:cNS=-2.0,\
        isr_G2QQ_cNS_up isr:G2QQ:cNS=2.0,\
        isr_Q2QG_cNS_dn isr:Q2QG:cNS=-2.0,\
        isr_Q2QG_cNS_up isr:Q2QG:cNS=2.0,\
        isr_X2XG_cNS_dn isr:X2XG:cNS=-2.0,\
        isr_X2XG_cNS_up isr:X2XG:cNS=2.0}");
        
    pythia.readString("UncertaintyBands:nFlavQ = 4"); // define X=bottom/top in X2XG variations
    pythia.readString("UncertaintyBands:MPIshowers = on");
    pythia.readString("UncertaintyBands:overSampleFSR = 10.0");
    pythia.readString("UncertaintyBands:overSampleISR = 10.0");
    pythia.readString("UncertaintyBands:FSRpTmin2Fac = 20");
    pythia.readString("UncertaintyBands:ISRpTmin2Fac = 20"); // for consistency with UL and P8.240 set to 20, to be optimized and changed for Run 3 re-MC  

    // Initialize Pythia
    pythia.init();

    // Set up interface to HepMC2
    HepMC::Pythia8ToHepMC toHepMC;
    HepMC::IO_GenEvent ascii_io(outFile, std::ios::out);

    // Event loop
    for (int i = 0; i < nEvent; ++i) {

        // Generate next event; if failure, skip
        if (!pythia.next()) {
            std::cerr << "Pythia event generation failed at event " << i << "\n";
            continue;
        }

        // Apply Z' -> mu+ mu- filters (pT/eta cuts on daughters)
        if (!passZPrimeMuMu(pythia.event)) {
            continue;
        }

        // Create a new HepMC event
        HepMC::GenEvent* hepmcevt = new HepMC::GenEvent(HepMC::Units::GEV, HepMC::Units::MM);

        // Fill the HepMC event from Pythia
        toHepMC.fill_next_event(pythia, hepmcevt);

        // Write event to file and clean up
        ascii_io << hepmcevt;
        delete hepmcevt;
    }

  // Print out statistics
  pythia.stat();

  return 0;
}
