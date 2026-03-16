///////////////////////////////////////////////////////////////
// delphes_card_verifier.cpp
//
// MadAnalysis5 analyzer to verify a Delphes card by measuring:
//   1. Muon tracking efficiency   (per eta-pT bin)
//   2. Momentum resolution        (mean and sigma per eta-pT bin)
//   3. Muon reconstruction efficiency (per eta-pT bin)
//
// Each metric is binned to match the Delphes card bin structure
// and written to separate CSV files at the end of the run.
//
// Output files:
//   tracking_efficiency.csv
//   momentum_resolution.csv
//   reco_efficiency.csv
//
// Usage: drop this file in as your MA5 user analyzer (user.cpp)
///////////////////////////////////////////////////////////////

#include "SampleAnalyzer/User/Analyzer/user.h"
using namespace MA5;
using namespace std;

#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <numeric>

// ---------------------------------------------------------------
// Bin definitions — chosen to match the Delphes card exactly
// ---------------------------------------------------------------

// Eta bin edges (muon tracking / reco efficiency)
static const std::vector<double> ETA_EDGES = {0.0, 0.9, 1.2, 2.1, 2.4};

// pT bin edges (GeV) — covers the J/psi range and beyond
static const std::vector<double> PT_EDGES = {
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 100.0
};

// Separate finer eta bins for momentum resolution (matches smearing card)
static const std::vector<double> RES_ETA_EDGES = {0.0, 0.5, 1.5, 2.5};

// ---------------------------------------------------------------
// Helper: find bin index (-1 if out of range)
// ---------------------------------------------------------------
static int findBin(double val, const std::vector<double>& edges)
{
    if (val < edges.front() || val >= edges.back()) return -1;
    for (int i = 0; i < (int)edges.size() - 1; ++i)
        if (val >= edges[i] && val < edges[i + 1]) return i;
    return -1;
}

// ---------------------------------------------------------------
// Per-bin statistics accumulator
// ---------------------------------------------------------------
struct BinStats
{
    long long n_gen    = 0;  // gen-level muons in this bin
    long long n_track  = 0;  // tracks matched to gen muons (tracking eff numerator)
    long long n_reco   = 0;  // reco muons matched to gen (reco eff numerator)
    double    sum_res  = 0.0; // sum of (pT_reco - pT_gen)/pT_gen
    double    sum_res2 = 0.0; // sum of squares
    long long n_res    = 0;  // number of matched track-gen pairs contributing to resolution
};

// Key: (eta_bin_index, pt_bin_index)
using BinKey = std::pair<int,int>;
using BinMap = std::map<BinKey, BinStats>;

// ---------------------------------------------------------------
// Phi normalization / deltaR utilities
// ---------------------------------------------------------------
static double normalizePhi(double phi)
{
    while (phi >  M_PI) phi -= 2.0 * M_PI;
    while (phi <= -M_PI) phi += 2.0 * M_PI;
    return phi;
}

static double deltaR(double eta1, double phi1, double eta2, double phi2)
{
    double dphi = normalizePhi(phi1 - phi2);
    double deta = eta1 - eta2;
    return std::sqrt(deta * deta + dphi * dphi);
}

// ---------------------------------------------------------------
// Output file handles
// ---------------------------------------------------------------
static std::ofstream trk_eff_csv;
static std::ofstream mom_res_csv;
static std::ofstream reco_eff_csv;

// Bin accumulators (indexed on gen-pT / gen-eta)
static BinMap g_eff_bins;      // tracking & reco efficiency (ETA_EDGES x PT_EDGES)
static BinMap g_res_bins;      // momentum resolution       (RES_ETA_EDGES x PT_EDGES)

///////////////////////////////////////////////////////////////
//                        Initialize                         //
///////////////////////////////////////////////////////////////
MAbool user::Initialize(const MA5::Configuration& cfg,
                        const std::map<std::string,std::string>& parameters)
{
    PHYSICS->recConfig().Reset();

    Manager()->AddRegionSelection("SR");

    // Open output CSV files
    trk_eff_csv.open("tracking_efficiency.csv");
    trk_eff_csv << "eta_lo,eta_hi,pt_lo,pt_hi,n_gen,n_track,tracking_efficiency\n";

    mom_res_csv.open("momentum_resolution.csv");
    mom_res_csv << "eta_lo,eta_hi,pt_lo,pt_hi,n_pairs,mean_resolution,sigma_resolution\n";

    reco_eff_csv.open("reco_efficiency.csv");
    reco_eff_csv << "eta_lo,eta_hi,pt_lo,pt_hi,n_gen,n_reco,reco_efficiency\n";

    // Pre-populate maps so every bin exists even if empty
    for (int ieta = 0; ieta < (int)ETA_EDGES.size() - 1; ++ieta)
        for (int ipt = 0; ipt < (int)PT_EDGES.size() - 1; ++ipt)
            g_eff_bins[{ieta, ipt}] = BinStats{};

    for (int ieta = 0; ieta < (int)RES_ETA_EDGES.size() - 1; ++ieta)
        for (int ipt = 0; ipt < (int)PT_EDGES.size() - 1; ++ipt)
            g_res_bins[{ieta, ipt}] = BinStats{};

    return true;
}

///////////////////////////////////////////////////////////////
//                         Execute                           //
///////////////////////////////////////////////////////////////
bool user::Execute(SampleFormat& sample, const EventFormat& event)
{
    double weight = 1.0;
    if (!Configuration().IsNoEventWeight() && event.mc() != nullptr)
        weight = event.mc()->weight();
    Manager()->InitializeForNewEvent(weight);

    if (event.rec() == nullptr || event.mc() == nullptr) return true;

    // ----------------------------------------------------------
    // STEP 1: Collect gen-level muons (PDG 13) within acceptance
    // ----------------------------------------------------------
    // We'll store pointers so we can match tracks/reco back to them
    std::vector<const MCParticleFormat*> gen_muons;
    for (const auto& part : event.mc()->particles())
    {
        if (std::abs(part.pdgid()) != 13)    continue;
        if (std::abs(part.eta())   >= 2.4)   continue;  // card acceptance
        if (part.pt()              <  1.0)   continue;  // card threshold
        gen_muons.push_back(&part);
    }

    // Increment gen counters (binned on gen eta/pT)
    for (const auto* gmu : gen_muons)
    {
        double abseta = std::abs(gmu->eta());
        double pt     = gmu->pt();

        int ieta_eff = findBin(abseta, ETA_EDGES);
        int ipt      = findBin(pt,     PT_EDGES);
        if (ieta_eff >= 0 && ipt >= 0)
            g_eff_bins[{ieta_eff, ipt}].n_gen++;

        int ieta_res = findBin(abseta, RES_ETA_EDGES);
        if (ieta_res >= 0 && ipt >= 0)
            g_res_bins[{ieta_res, ipt}].n_gen++;
    }

    // ----------------------------------------------------------
    // STEP 2: Tracking efficiency + momentum resolution
    //
    // A track with |pdgid|=13 that has a valid mc() pointer has
    // passed TrackingEfficiency and MomentumSmearing in Delphes.
    // We use gen (mc()) quantities for the bin lookup so we're
    // measuring efficiency as a function of truth kinematics.
    // ----------------------------------------------------------
    for (const auto& trk : event.rec()->tracks())
    {
        if (std::abs(trk.pdgid()) != 13) continue;
        if (trk.mc() == nullptr)         continue;

        double gen_abseta = std::abs(trk.mc()->eta());
        double gen_pt     = trk.mc()->pt();
        double reco_pt    = trk.pt();

        if (gen_pt <= 0.0) continue;

        // --- Tracking efficiency numerator ---
        int ieta_eff = findBin(gen_abseta, ETA_EDGES);
        int ipt      = findBin(gen_pt,     PT_EDGES);
        if (ieta_eff >= 0 && ipt >= 0)
            g_eff_bins[{ieta_eff, ipt}].n_track++;

        // --- Momentum resolution ---
        double res = (reco_pt - gen_pt) / gen_pt;
        int ieta_res = findBin(gen_abseta, RES_ETA_EDGES);
        if (ieta_res >= 0 && ipt >= 0)
        {
            auto& b = g_res_bins[{ieta_res, ipt}];
            b.sum_res  += res;
            b.sum_res2 += res * res;
            b.n_res++;
        }
    }

    // ----------------------------------------------------------
    // STEP 3: Reconstruction efficiency
    //
    // event.rec()->muons() returns objects that have passed
    // MuonEfficiency (the ID step after smearing).
    // We match each reco muon back to the nearest gen muon within
    // deltaR < 0.1 to count the reco efficiency numerator.
    //
    // NOTE: For low-pT J/psi muons the standard muon collection
    // may be empty (MuonEfficiency threshold at pT>3 GeV in barrel).
    // In that regime, track-based muons dominate; the reco efficiency
    // measured here reflects the combined tracking+ID efficiency.
    // ----------------------------------------------------------
    std::vector<bool> gen_matched(gen_muons.size(), false);

    for (const auto& rmu : event.rec()->muons())
    {
        double reco_abseta = std::abs(rmu.eta());
        double reco_pt     = rmu.pt();

        if (reco_abseta >= 2.4) continue;

        // Match to closest gen muon
        double best_dr   = 0.1; // matching cone
        int    best_idx  = -1;

        for (int i = 0; i < (int)gen_muons.size(); ++i)
        {
            if (gen_matched[i]) continue;
            double dr = deltaR(gen_muons[i]->eta(), gen_muons[i]->phi(),
                               rmu.eta(),           rmu.phi());
            if (dr < best_dr)
            {
                best_dr  = dr;
                best_idx = i;
            }
        }

        if (best_idx < 0) continue;
        gen_matched[best_idx] = true;

        // Use gen kinematics for bin lookup
        double gen_abseta = std::abs(gen_muons[best_idx]->eta());
        double gen_pt     = gen_muons[best_idx]->pt();

        int ieta_eff = findBin(gen_abseta, ETA_EDGES);
        int ipt      = findBin(gen_pt,     PT_EDGES);
        if (ieta_eff >= 0 && ipt >= 0)
            g_eff_bins[{ieta_eff, ipt}].n_reco++;
    }

    return true;
}

///////////////////////////////////////////////////////////////
//                        Finalize                           //
///////////////////////////////////////////////////////////////
void user::Finalize(const SampleFormat& summary,
                    const std::vector<SampleFormat>& files)
{
    // ----------------------------------------------------------
    // Write tracking efficiency and reco efficiency
    // ----------------------------------------------------------
    for (int ieta = 0; ieta < (int)ETA_EDGES.size() - 1; ++ieta)
    {
        for (int ipt = 0; ipt < (int)PT_EDGES.size() - 1; ++ipt)
        {
            const auto& b = g_eff_bins.at({ieta, ipt});

            double trk_eff  = (b.n_gen > 0) ? (double)b.n_track / b.n_gen : 0.0;
            double reco_eff = (b.n_gen > 0) ? (double)b.n_reco  / b.n_gen : 0.0;

            trk_eff_csv
                << ETA_EDGES[ieta]   << "," << ETA_EDGES[ieta+1] << ","
                << PT_EDGES[ipt]     << "," << PT_EDGES[ipt+1]   << ","
                << b.n_gen           << "," << b.n_track          << ","
                << trk_eff           << "\n";

            reco_eff_csv
                << ETA_EDGES[ieta]   << "," << ETA_EDGES[ieta+1] << ","
                << PT_EDGES[ipt]     << "," << PT_EDGES[ipt+1]   << ","
                << b.n_gen           << "," << b.n_reco           << ","
                << reco_eff          << "\n";
        }
    }

    // ----------------------------------------------------------
    // Write momentum resolution
    // mean  = E[res]           = sum_res / n_res
    // sigma = sqrt(E[res^2] - E[res]^2)
    // ----------------------------------------------------------
    for (int ieta = 0; ieta < (int)RES_ETA_EDGES.size() - 1; ++ieta)
    {
        for (int ipt = 0; ipt < (int)PT_EDGES.size() - 1; ++ipt)
        {
            const auto& b = g_res_bins.at({ieta, ipt});

            double mean  = 0.0;
            double sigma = 0.0;

            if (b.n_res > 1)
            {
                mean        = b.sum_res / b.n_res;
                double var  = b.sum_res2 / b.n_res - mean * mean;
                sigma       = (var > 0.0) ? std::sqrt(var) : 0.0;
            }

            mom_res_csv
                << RES_ETA_EDGES[ieta]   << "," << RES_ETA_EDGES[ieta+1] << ","
                << PT_EDGES[ipt]         << "," << PT_EDGES[ipt+1]        << ","
                << b.n_res               << ","
                << mean                  << ","
                << sigma                 << "\n";
        }
    }

    trk_eff_csv.close();
    mom_res_csv.close();
    reco_eff_csv.close();
}
