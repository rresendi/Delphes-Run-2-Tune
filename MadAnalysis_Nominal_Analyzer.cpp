#include "SampleAnalyzer/User/Analyzer/user.h"
using namespace MA5;
using namespace std;
#include <cstdlib>
#include <tuple>

template <typename T>
T normalizePhi(T phi)
{
  while (phi > T(M_PI))
    phi -= T(2 * M_PI);
  while (phi <= -T(M_PI))
    phi += T(2 * M_PI);
  return phi;
}

template <typename T, typename U>
T computeDeltaPhi(T phi1, U phi2)
{
  phi1 = normalizePhi(phi1);
  phi2 = normalizePhi(phi2);
  T dphi = phi1 - phi2;
  while (dphi > T(M_PI))
    dphi -= T(2 * M_PI);
  while (dphi <= -T(M_PI))
    dphi += T(2 * M_PI);
  return dphi;
}

template <typename T, typename U>
double calculateDeltaR(const T &obj1, const U &obj2)
{
  double eta1 = obj1.eta();
  double eta2 = obj2.eta();

  double dphi = std::fabs(computeDeltaPhi(obj1.phi(), obj2.phi()));
  double deta = eta1 - eta2;

  return std::sqrt(deta * deta + dphi * dphi);
}

// Returns the mass (in GeV) for a charged particle based on its PDG id
// Common charged leptons, mesons, and baryons are included
// If the PDG id is not recognized, defaults to the charged pion mass
double getMassFromPDG(int pdgid)
{
  int absid = std::abs(pdgid);
  switch (absid)
  {
  case 13:
    return 0.1056583745; // muon
  default:
  {
    std::cout << "[FATAL] Unrecognized PDG id in getMassFromPDG: " + std::to_string(pdgid) + " - please update the switch function in getMassFromPDG with this particle's PDG ID and mass (Gev)." << std::endl;
    throw std::runtime_error("[FATAL] Unrecognized PDG id in getMassFromPDG: " + std::to_string(pdgid) + " - please update the switch function in getMassFromPDG with this particle's PDG ID and mass (Gev).");
  }
  }
}

bool compareBypT(const RecTrackFormat &a, const RecTrackFormat &b)
{
  return a.pt() > b.pt(); // Sort by pT in descending order
}

// Function to sort various lepton collections by pT
void sortLeptonCollections(std::vector<RecTrackFormat> &muons,
                           std::vector<RecTrackFormat> &posMuons,
                           std::vector<RecTrackFormat> &negMuons)
{

  // Sorting muons
  std::sort(muons.begin(), muons.end(), compareBypT);
  std::sort(posMuons.begin(), posMuons.end(), compareBypT);
  std::sort(negMuons.begin(), negMuons.end(), compareBypT);
}

vector<RecTrackFormat> filter_muons(vector<RecTrackFormat> objects,
                                     float ptmin,
                                     float ptmax,
                                     float etamax,
                                     int pdgid,
                                     std::string charge = "")
{
  vector<RecTrackFormat> filtered;
  for (auto &obj : objects)
  {
    if (std::fabs(obj.pdgid()) != pdgid) continue;

    // kinematic selections
    if (obj.pt() < ptmin)       continue;
    if (obj.pt() > ptmax)       continue;
    if (std::fabs(obj.eta()) > etamax) continue;
    if (charge == "+")
    {
      if (obj.pdgid() < 0) continue;
    }
    else if (charge == "-")
    {
      if (obj.pdgid() > 0) continue;
    }

    filtered.push_back(obj);
  }

  return filtered;
}

///////////////////////////////////////////////////////////////
//                       Initialize                          //
// Function called one time at the beginning of the analysis //
///////////////////////////////////////////////////////////////

MAbool user::Initialize(const MA5::Configuration &cfg,
                        const std::map<std::string, std::string> &parameters)
{
  
  // Initializing PhysicsService for RECO
  PHYSICS->recConfig().Reset();

  // Define a region
  Manager()->AddRegionSelection("SR");

  // ===== Selections ===== //
  Manager()->AddCut("twoOSleptons");

  // ===== Histograms ===== //
  // JPSI Histograms
  Manager()->AddHisto("JPSIMass", 140, 2.9, 3.3);
  Manager()->AddHisto("JPSIpT", 200, 0, 200);
  Manager()->AddHisto("JPSIETA", 20, -2.5, 2.5);
  Manager()->AddHisto("JPSIPHI", 20, -3.14, 3.14);
  // Muon Histograms
  Manager()->AddHisto("LeadMuonPT", 100, 0.0, 100.0);
  Manager()->AddHisto("SubleadMuonPT", 100, 0.0, 100);
  Manager()->AddHisto("LeadMuonETA", 20, -2.5, 2.5);
  Manager()->AddHisto("SubleadMuonETA", 20, -2.5, 2.5);
  Manager()->AddHisto("LeadMuonPHI", 20, -3.14, 3.14);
  Manager()->AddHisto("SubleadMuonPHI", 20, -3.14, 3.14);

  return true;
}

///////////////////////////////////////////////////////////////
//                          Execute                          //
//        Function called each time an event is read         //
///////////////////////////////////////////////////////////////

bool user::Execute(SampleFormat &sample, const EventFormat &event)
{
  
  // Event weight
  double weight = 1.;
  if (!Configuration().IsNoEventWeight() && event.mc() != 0)
  {
    weight = event.mc()->weight();
  }
  Manager()->InitializeForNewEvent(weight);
  if (event.rec() == 0)
  {
    return true;
  }

  // MUONS
  float const MUON_PT_MIN = 0;
  float const MUON_PT_MAX = 10;
  float const MUON_ETA_MAX = 2.4;
  int PDGID = 13;

  //////////////////////////////////////////////
  //  Applying Base Lepton Object Selections  //
  //////////////////////////////////////////////

  // Muon Collections - use tracks for low pT Jpsi
  vector<RecTrackFormat> muons = filter_muons(event.rec()->tracks(), MUON_PT_MIN, MUON_PT_MAX, MUON_ETA_MAX, PDGID);
  vector<RecTrackFormat> posMuons = filter_muons(muons, MUON_PT_MIN,MUON_PT_MAX, MUON_ETA_MAX, PDGID, "+");
  vector<RecTrackFormat> negMuons = filter_muons(muons, MUON_PT_MIN,MUON_PT_MAX, MUON_ETA_MAX, PDGID, "-");

  // Sort all collections before event selection
  sortLeptonCollections(muons, posMuons, negMuons);

  //////////////////////////////////////////////
  //           Event level selections         //
  //////////////////////////////////////////////

  // Event Cut definitions

  // Two OSSF Lepton + Lead Lepton pT Selection
  bool twoOSleptons = (posMuons.size() > 0 && negMuons.size() > 0);              // Require exactly two opposite-sign leptons (either muons or electrons)
  if (not Manager()->ApplyCut(twoOSleptons, "twoOSleptons")) {
    return true;
  }

  // Reconstruct the JPSI

  // At this point there are exactly one positive and one negative lepton
  RecTrackFormat posMuon = posMuons.at(0);
  RecTrackFormat negMuon = negMuons.at(0);
  ParticleBaseFormat recoJPSI;
  recoJPSI += posMuon.momentum();
  recoJPSI += negMuon.momentum();

  //////////////////////////////////////////////
  //           Filling Histograms             //
  //////////////////////////////////////////////

  // JPSI Histograms
  Manager()->FillHisto("JPSIMass",   recoJPSI.m());
  Manager()->FillHisto("JPSIpT",     recoJPSI.pt());
  Manager()->FillHisto("JPSIETA",    recoJPSI.eta());
  Manager()->FillHisto("JPSIPHI",    recoJPSI.phi());

  // Muon or Electron Histograms
  if (muons.size() > 0)
  {
    Manager()->FillHisto("LeadMuonPT", muons.at(0).pt());
    Manager()->FillHisto("LeadMuonETA", muons.at(0).eta());
    Manager()->FillHisto("LeadMuonPHI", muons.at(0).phi());
  }
    if (muons.size() > 1)
  {
    Manager()->FillHisto("SubleadMuonPT", muons.at(1).pt());
    Manager()->FillHisto("SubleadMuonETA", muons.at(1).eta());
    Manager()->FillHisto("SubleadMuonPHI", muons.at(1).phi());
  }

  return true;
}

///////////////////////////////////////////////////////////////
//                        Finalize                           //
//    Function called one time at the end of the analysis    //
///////////////////////////////////////////////////////////////
void user::Finalize(const SampleFormat &summary, const std::vector<SampleFormat> &files)
{
}
