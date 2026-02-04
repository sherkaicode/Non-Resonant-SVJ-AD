/*
Analyzer.C
Matches logic of the Python/uproot reference script, 
with a switch to enable/disable cuts.
Updated to include pt_balance.
*/

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include "TVector2.h"
#include <algorithm>
#include <vector>
#else
class ExRootTreeReader;
#endif

//------------------------------------------------------------------------------

void AnalyseEvents(ExRootTreeReader *treeReader, const char *outputFile, bool applyCuts)
{
  // --- Setup Branches ---
  TClonesArray *branchEvent     = treeReader->UseBranch("Event");
  TClonesArray *branchFatJet    = treeReader->UseBranch("FatJet");
  TClonesArray *branchSmallJet  = treeReader->UseBranch("Jet");
  TClonesArray *branchMET       = treeReader->UseBranch("MissingET");
  TClonesArray *branchElectron  = treeReader->UseBranch("Electron");
  TClonesArray *branchMuon      = treeReader->UseBranch("Muon");

  Long64_t allEntries = treeReader->GetEntries();
  ofstream fout(outputFile);

  cout << "** Chain contains " << allEntries << " events" << endl;
  cout << "** Apply Cuts: " << (applyCuts ? "YES" : "NO") << endl;

  // Added pt_balance to header
  fout << "pT_j1 eta_j1 phi_j1 pT_j2 eta_j2 phi_j2 m_jj sm_jj "
       << "tau21_j1 tau21_j2 tau32_j1 tau32_j2 "
       << "met phi_met min_dPhi ht pt_balance delta_phi_j1j2 weight" << endl;

  int nSelected = 0;

  for(Int_t entry = 0; entry < allEntries; ++entry)
  {
    treeReader->ReadEntry(entry);
    if(entry % 1000 == 0) cout << "Processing event: " << entry << endl;

    int nSmallJets = branchSmallJet->GetEntries();
    int nFatJets   = branchFatJet->GetEntries();
    
    // Pointers to objects (initially nullptr)
    Jet *j1  = (nSmallJets > 0) ? (Jet*) branchSmallJet->At(0) : nullptr;
    Jet *j2  = (nSmallJets > 1) ? (Jet*) branchSmallJet->At(1) : nullptr;
    MissingET *metObj = (MissingET*) branchMET->At(0);

    // --- Pre-calculate physics variables needed for cuts ---
    
    double phi_met = metObj->Phi;
    
    // Calculate dPhis for all small jets and find closest/farthest for pt_balance
    std::vector<double> dphis;
    int nLowDPhi = 0; 

    // Variables for pt_balance logic
    Jet *jetClosest = nullptr;
    Jet *jetFarthest = nullptr;
    double minAbsDPhi = 9999.0;
    double maxAbsDPhi = -1.0;

    for(int i=0; i < nSmallJets; i++) {
        Jet *jet = (Jet*) branchSmallJet->At(i);
        double dphi = fabs(TVector2::Phi_mpi_pi(jet->Phi - phi_met));
        dphis.push_back(dphi);
        if (dphi < 2.0) nLowDPhi++;

        // Logic to find closest and farthest jet from MET
        if (dphi < minAbsDPhi) {
            minAbsDPhi = dphi;
            jetClosest = jet;
        }
        if (dphi > maxAbsDPhi) {
            maxAbsDPhi = dphi;
            jetFarthest = jet;
        }
    }

    // Count Central Jets
    int nCentral = 0;
    for(int i=0; i < nSmallJets; i++) {
        Jet *jet = (Jet*) branchSmallJet->At(i);
        if(fabs(jet->Eta) < 2.8) nCentral++;
    }

    // Count B-Tags
    int nBTags = 0;
    for(int i=0; i < nSmallJets; i++) {
        Jet *jet = (Jet*) branchSmallJet->At(i);
        if(jet->BTag == 1) nBTags++;
    }

    // Count Taus
    int nTaus = 0;
    for(int i=0; i < nSmallJets; i++) {
        Jet *jet = (Jet*) branchSmallJet->At(i);
        if(jet->TauTag == 1) nTaus++;
    }


    // ==========================================================
    //                        CUT LOGIC
    // ==========================================================
    if (applyCuts) {
        // 1. At least 2 Small Jets
        if (nSmallJets < 2) continue;

        // 2. At least 2 Central Jets
        if (nCentral < 2) continue;

        // 3. Leading Jet pT > 250
        if (j1->PT < 250.0) continue;

        // 4. Subleading Jet pT > 30
        if (j2->PT < 30.0) continue;

        // 5. dPhi Logic (At least 1 jets with dPhi < 2.0 / QCD-like)
        if (nLowDPhi == 0) continue;

        // 6. B-Tag Veto
        if (nBTags >= 2) continue;

        // 7. Tau Veto
        if (nTaus > 0) continue;

        // // 8. Electron Veto
        // if (branchElectron->GetEntries() > 0) continue;

        // // 9. Muon Veto
        // if (branchMuon->GetEntries() > 0) continue;

        // 10. At least 2 Fat Jets
        // if (nFatJets < 2) continue;

        if (metObj->MET < 250.0) continue;

        // 8. Electron Veto (No electrons with pT > 7 GeV)
        bool failElectron = false;
        for(int i=0; i < branchElectron->GetEntries(); ++i) {
            Electron *ele = (Electron*) branchElectron->At(i);
            if(ele->PT > 7.0) { failElectron = true; break; }
        }
        if (failElectron) continue;

        // 9. Muon Veto (No muons with pT > 7 GeV)
        bool failMuon = false;
        for(int i=0; i < branchMuon->GetEntries(); ++i) {
            Muon *mu = (Muon*) branchMuon->At(i);
            if(mu->PT > 7.0) { failMuon = true; break; }
        }
        if (failMuon) continue;

        // 10. SR Kinematics: MET > 600 GeV
        if (metObj->MET < 600.0) continue;

        // 11. SR Kinematics: HT > 600 GeV
        // (HT is calculated as the scalar sum of pT of all small jets)
        double ht_val = 0.0;
        for(int i=0; i < nSmallJets; i++) {
            Jet *jet = (Jet*) branchSmallJet->At(i);
            ht_val += jet->PT;
        }
        if (ht_val < 600.0) continue;
        
    }
    // ==========================================================


    // --- Retrieve FatJets safely (check existence) ---
    Jet *fj1 = (nFatJets > 0) ? (Jet*) branchFatJet->At(0) : nullptr;
    Jet *fj2 = (nFatJets > 1) ? (Jet*) branchFatJet->At(1) : nullptr;

    // --- Calculate Output Variables ---

    // M_jj (Safe)
    double m_jj = (fj1 && fj2) ? (fj1->P4() + fj2->P4()).M() : -1.0;
    double sm_jj = (j1 && j2) ? (j1->P4() + j2->P4()).M() : -1.0;

    // Tau Ratios (Safe)
    double tau21_j1 = (fj1 && fj1->Tau[0] > 0) ? fj1->Tau[1] / fj1->Tau[0] : -1.0;
    double tau21_j2 = (fj2 && fj2->Tau[0] > 0) ? fj2->Tau[1] / fj2->Tau[0] : -1.0;
    double tau32_j1 = (fj1 && fj1->Tau[1] > 0) ? fj1->Tau[2] / fj1->Tau[1] : -1.0;
    double tau32_j2 = (fj2 && fj2->Tau[1] > 0) ? fj2->Tau[2] / fj2->Tau[1] : -1.0;

    // min_dPhi
    double min_dPhi = 999.0;
    if (dphis.size() > 0) {
        for(double d : dphis) {
            if(d < min_dPhi) min_dPhi = d;
        }
    } else {
        min_dPhi = -1.0;
    }

    // HT
    double ht = 0.0;
    for(int i=0; i < nSmallJets; i++) {
        Jet *jet = (Jet*) branchSmallJet->At(i);
        ht += jet->PT;
    }

    // --- Calculate pt_balance ---
    double pt_balance = -1.0;
    double delta_phi_j1j2 = -1.0;

    if (jetClosest && jetFarthest) {
        // Construct pT vectors
        TVector2 vClose, vFar;
        vClose.SetMagPhi(jetClosest->PT, jetClosest->Phi);
        vFar.SetMagPhi(jetFarthest->PT, jetFarthest->Phi);

        // Numerator: Magnitude of Vector Sum |pT1 + pT2|
        double num = (vClose + vFar).Mod();

        // Denominator: Scalar Sum |pT1| + |pT2|
        double den = jetClosest->PT + jetFarthest->PT;

        if (den > 0) pt_balance = num / den;

        delta_phi_j1j2 = fabs(TVector2::Phi_mpi_pi(jetClosest->Phi - jetFarthest->Phi));
    }

    HepMCEvent *ev = (HepMCEvent*) branchEvent->At(0);
    double weight = ev->Weight;
    double met_val = metObj->MET;


    // --- Write to File (using Ternary operators for safety) ---
    fout << (fj1 ? fj1->PT : -1)  << " " << (fj1 ? fj1->Eta : -1) << " " << (fj1 ? fj1->Phi : -1) << " "
         << (fj2 ? fj2->PT : -1)  << " " << (fj2 ? fj2->Eta : -1) << " " << (fj2 ? fj2->Phi : -1) << " "
         << m_jj << " " << sm_jj << " "
         << tau21_j1 << " " << tau21_j2 << " "
         << tau32_j1 << " " << tau32_j2 << " "
         << met_val << " " << phi_met << " "
         << min_dPhi << " " << ht << " " 
         << pt_balance << " " // Added pt_balance here
         << delta_phi_j1j2 << " "
         << weight
         << endl;

    nSelected++;
  }

  cout << "** Selected events: " << nSelected << " / " << allEntries << endl;
  fout.close();
}

//------------------------------------------------------------------------------

void Analyzer(const char *inputFile, const char *outputFile)
{
  gSystem->Load("libDelphes");

  TChain *chain = new TChain("Delphes");
  chain->Add(inputFile);

  ExRootTreeReader *treeReader = new ExRootTreeReader(chain);

  // ----------------------------------------------------
  // SWITCH: Set this to true (apply cuts) or false (save all)
  // ----------------------------------------------------
  bool applyCuts = true; 
  
  AnalyseEvents(treeReader, outputFile, applyCuts);

  delete treeReader;
  delete chain;
}