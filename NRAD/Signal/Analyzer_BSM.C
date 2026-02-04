/*
Analyzer_NoOverlap.C
BSM Analysis:
- "Invisible Muon" MET Recalculation (Treats Muons as invisible).
- NO Overlap Removal (Jets and Electrons are kept as-is).
- Standard Signal Region Cuts enabled.
*/

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include "TVector2.h"
#include "TMath.h"
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#else
class ExRootTreeReader;
#endif

// --- Helper: DeltaR Calculation ---
double CalculateDeltaR(double eta1, double phi1, double eta2, double phi2) {
    double dEta = eta1 - eta2;
    double dPhi = TVector2::Phi_mpi_pi(phi1 - phi2);
    return TMath::Sqrt(dEta*dEta + dPhi*dPhi);
}

//------------------------------------------------------------------------------

void AnalyseEvents(ExRootTreeReader *treeReader, const char *outputFile, bool applyCuts)
{
  TClonesArray *branchEvent     = treeReader->UseBranch("Event");
  TClonesArray *branchFatJet    = treeReader->UseBranch("FatJet");
  TClonesArray *branchSmallJet  = treeReader->UseBranch("Jet");
  TClonesArray *branchElectron  = treeReader->UseBranch("Electron");
  TClonesArray *branchMuon      = treeReader->UseBranch("Muon");
  TClonesArray *branchTrack     = treeReader->UseBranch("Track");

  Long64_t allEntries = treeReader->GetEntries();
  ofstream fout(outputFile);

  cout << "** Chain contains " << allEntries << " events" << endl;

  // Header matching your requested format
  fout << "pT_j1 eta_j1 phi_j1 pT_j2 eta_j2 phi_j2 m_jj sm_jj "
       << "tau21_j1 tau21_j2 tau32_j1 tau32_j2 "
       << "met phi_met min_dPhi ht pt_balance delta_phi_j1j2 weight" << endl;

  int nSelected = 0;

  for(Int_t entry = 0; entry < allEntries; ++entry)
  {
    treeReader->ReadEntry(entry);
    if(entry % 5000 == 0) cout << "Processing event: " << entry << endl;

    // ----------------------------------------------------
    // 1. OBJECT SELECTION (Direct from branches, no cleaning)
    // ----------------------------------------------------
    
    // Select Jets (pT > 30)
    std::vector<Jet*> selectedJets;
    for(int i=0; i < branchSmallJet->GetEntries(); ++i) {
        Jet *jet = (Jet*) branchSmallJet->At(i);
        if(jet->PT > 30.0) selectedJets.push_back(jet);
    }

    // Select Electrons (pT > 7, |eta| < 2.5)
    std::vector<Electron*> selectedElectrons;
    for(int i=0; i < branchElectron->GetEntries(); ++i) {
        Electron *ele = (Electron*) branchElectron->At(i);
        if(ele->PT > 7.0 && fabs(ele->Eta) < 2.5) selectedElectrons.push_back(ele);
    }

    // Select Muons (We need them ONLY to remove their tracks from MET)
    std::vector<Muon*> allMuons;
    for(int i=0; i < branchMuon->GetEntries(); ++i) {
        Muon *mu = (Muon*) branchMuon->At(i);
        if(mu->PT > 7.0) allMuons.push_back(mu);
    }

    // ----------------------------------------------------
    // 2. RECONSTRUCT MET (Invisible Muon Logic)
    // ----------------------------------------------------
    
    double sumPx = 0.0;
    double sumPy = 0.0;

    // A. Add Electrons
    for(Electron* ele : selectedElectrons) {
        sumPx += ele->PT * TMath::Cos(ele->Phi);
        sumPy += ele->PT * TMath::Sin(ele->Phi);
    }

    // B. Add Jets (Note: Potential double counting of electrons here, as requested)
    for(Jet* jet : selectedJets) {
        sumPx += jet->PT * TMath::Cos(jet->Phi);
        sumPy += jet->PT * TMath::Sin(jet->Phi);
    }

    // C. Add Soft Term (Tracks not matched to anything)
    // We removed the strict PV requirement to ensure safety for all file types
    for(int i = 0; i < branchTrack->GetEntries(); ++i) {
        Track *trk = (Track*) branchTrack->At(i);
        
        bool matched = false;

        // Veto tracks from Electrons (dR < 0.05)
        for(Electron* ele : selectedElectrons) {
            if(CalculateDeltaR(trk->Eta, trk->Phi, ele->Eta, ele->Phi) < 0.05) { matched = true; break; }
        }
        if(matched) continue;

        // Veto tracks from Jets (dR < 0.4)
        for(Jet* jet : selectedJets) {
             if(CalculateDeltaR(trk->Eta, trk->Phi, jet->Eta, jet->Phi) < 0.4) { matched = true; break; }
        }
        if(matched) continue;

        // Veto tracks from Muons (dR < 0.05) -> CRITICAL for Invisible Muon Logic
        for(Muon* mu : allMuons) {
            if(CalculateDeltaR(trk->Eta, trk->Phi, mu->Eta, mu->Phi) < 0.05) { matched = true; break; }
        }
        if(matched) continue;

        // Add unmatched track momentum
        sumPx += trk->PT * TMath::Cos(trk->Phi);
        sumPy += trk->PT * TMath::Sin(trk->Phi);
    }

    double rec_met_val = TMath::Sqrt(sumPx*sumPx + sumPy*sumPy);
    double rec_met_phi = TMath::ATan2(-sumPy, -sumPx);

    // ----------------------------------------------------
    // 3. SIGNAL REGION CUTS
    // ----------------------------------------------------
    if (applyCuts) {
        // 1. At least 2 Jets
        if (selectedJets.size() < 2) continue;

        Jet *j1 = selectedJets[0];

        // 2. Leading Jet pT > 250 GeV
        if (j1->PT < 250.0) continue;

        // 3. MET > 250 GeV (Using our Recalculated MET)
        if (rec_met_val < 250.0) continue;

        // 4. Central Jet Requirement (>=2 jets with |eta| < 2.8)
        int nCentral = 0;
        for(Jet* jet : selectedJets) if(fabs(jet->Eta) < 2.8) nCentral++;
        if (nCentral < 2) continue;

        // 5. dPhi(Jet, MET) > 0.4 (Optional: To suppress QCD background)
        // Usually we cut if dPhi < 0.4 for *any* of the leading jets.
        // I will leave this *off* by default unless you specifically want QCD suppression.
        
        // 6. Lepton Veto? 
        // If your signal produces muons, DO NOT uncomment the muon veto!
        // if (selectedElectrons.size() > 0) continue;
    }

    // ----------------------------------------------------
    // 4. CALCULATE VARIABLES & WRITE OUTPUT
    // ----------------------------------------------------

    // --- Variables for Analysis ---
    double ht = 0.0;
    double min_dPhi = 999.0;
    Jet *jetClosest = nullptr;
    Jet *jetFarthest = nullptr;
    double minAbsDPhi = 9999.0;
    double maxAbsDPhi = -1.0;

    for(Jet* jet : selectedJets) {
        ht += jet->PT;
        double dphi = fabs(TVector2::Phi_mpi_pi(jet->Phi - rec_met_phi));
        if(dphi < min_dPhi) min_dPhi = dphi;

        if (dphi < minAbsDPhi) { minAbsDPhi = dphi; jetClosest = jet; }
        if (dphi > maxAbsDPhi) { maxAbsDPhi = dphi; jetFarthest = jet; }
    }
    if(selectedJets.empty()) min_dPhi = -1.0;

    double pt_balance = -1.0;
    double delta_phi_j1j2 = -1.0;

    if (jetClosest && jetFarthest) {
        TVector2 vClose, vFar;
        vClose.SetMagPhi(jetClosest->PT, jetClosest->Phi);
        vFar.SetMagPhi(jetFarthest->PT, jetFarthest->Phi);
        double den = jetClosest->PT + jetFarthest->PT;
        if (den > 0) pt_balance = (vClose + vFar).Mod() / den;
        delta_phi_j1j2 = fabs(TVector2::Phi_mpi_pi(jetClosest->Phi - jetFarthest->Phi));
    }

    // FatJet Variables
    Jet *fj1 = (branchFatJet->GetEntries() > 0) ? (Jet*) branchFatJet->At(0) : nullptr;
    Jet *fj2 = (branchFatJet->GetEntries() > 1) ? (Jet*) branchFatJet->At(1) : nullptr;
    
    // Small Jet Pointers
    Jet *sj1 = (selectedJets.size() > 0) ? selectedJets[0] : nullptr;
    Jet *sj2 = (selectedJets.size() > 1) ? selectedJets[1] : nullptr;

    HepMCEvent *ev = (HepMCEvent*) branchEvent->At(0);
    
    fout << (fj1 ? fj1->PT : -1)  << " " << (fj1 ? fj1->Eta : -1) << " " << (fj1 ? fj1->Phi : -1) << " "
         << (fj2 ? fj2->PT : -1)  << " " << (fj2 ? fj2->Eta : -1) << " " << (fj2 ? fj2->Phi : -1) << " "
         << (fj1 && fj2 ? (fj1->P4()+fj2->P4()).M() : -1) << " " 
         << (sj1 && sj2 ? (sj1->P4()+sj2->P4()).M() : -1) << " "
         << (fj1 && fj1->Tau[0]>0 ? fj1->Tau[1]/fj1->Tau[0] : -1) << " " 
         << (fj2 && fj2->Tau[0]>0 ? fj2->Tau[1]/fj2->Tau[0] : -1) << " "
         << (fj1 && fj1->Tau[1]>0 ? fj1->Tau[2]/fj1->Tau[1] : -1) << " " 
         << (fj2 && fj2->Tau[1]>0 ? fj2->Tau[2]/fj2->Tau[1] : -1) << " "
         << rec_met_val << " " << rec_met_phi << " "
         << min_dPhi << " " << ht << " " 
         << pt_balance << " " 
         << delta_phi_j1j2 << " "
         << ev->Weight
         << endl;

    nSelected++;
  }
  
  cout << "** Selected events: " << nSelected << " / " << allEntries << endl;
  fout.close();
}

void Analyzer_BSM(const char *inputFile, const char *outputFile)
{
  gSystem->Load("libDelphes");
  TChain *chain = new TChain("Delphes");
  chain->Add(inputFile);
  ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
  AnalyseEvents(treeReader, outputFile, true);
  delete treeReader;
  delete chain;
}