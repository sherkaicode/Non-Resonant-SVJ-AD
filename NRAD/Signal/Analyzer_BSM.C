/*
Analyzer_BSM.C
Full Analysis with Object Selection, Overlap Removal, Event Cuts, and "Invisible Muon" MET.

1. Vertex Selection: PV with highest sum pT^2 (>=2 tracks).
2. Object Selection:
   - Jets (pT > 30).
   - Ele/Mu (pT > 7, |eta| < 2.5).
3. Overlap Removal:
   - Cleaning Ele vs Mu, Jet vs Ele, Jet vs Ghost Mu, Ele/Mu vs Jet.
4. MET Reconstruction (Invisible Muon Modification):
   - Vector sum of Selected Jets + Selected Ele (Muons are EXCLUDED/Invisible).
   - PLUS tracks from PV not matched to Jets/Ele/Muons.
   - MET = - (Sum Px, Sum Py).
5. Event Cuts:
   - Lead Jet > 250, >=2 Central Jets.
   - dPhi(Jet, MET) < 2.0 using Recalculated MET.
   - Vetoes (B-Jet, Tau, Lepton).
*/

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include "TVector2.h"
#include "TMath.h"
#include <algorithm>
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
  // --- Setup Branches ---
  TClonesArray *branchEvent     = treeReader->UseBranch("Event");
  TClonesArray *branchFatJet    = treeReader->UseBranch("FatJet");
  TClonesArray *branchSmallJet  = treeReader->UseBranch("Jet");
  TClonesArray *branchMET       = treeReader->UseBranch("MissingET");
  TClonesArray *branchElectron  = treeReader->UseBranch("Electron");
  TClonesArray *branchMuon      = treeReader->UseBranch("Muon");
  TClonesArray *branchTrack     = treeReader->UseBranch("Track");

  Long64_t allEntries = treeReader->GetEntries();
  ofstream fout(outputFile);

  cout << "** Chain contains " << allEntries << " events" << endl;

  fout << "pT_j1 eta_j1 phi_j1 pT_j2 eta_j2 phi_j2 m_jj sm_jj "
       << "tau21_j1 tau21_j2 tau32_j1 tau32_j2 "
       << "met phi_met min_dPhi ht pt_balance delta_phi_j1j2 weight" << endl;

  int nSelected = 0;

  for(Int_t entry = 0; entry < allEntries; ++entry)
  {
    treeReader->ReadEntry(entry);
    if(entry % 1000 == 0) cout << "Processing event: " << entry << endl;

    // ==========================================================
    // 1. PRIMARY VERTEX SELECTION
    // ==========================================================
    std::map<int, std::pair<double, int>> vertexMap;
    int nTracks = branchTrack->GetEntries();

    for(int i = 0; i < nTracks; ++i) {
        Track *trk = (Track*) branchTrack->At(i);
        if(trk->PT > 0.5) { // 500 MeV
            int vIdx = trk->VertexIndex;
            if(vertexMap.find(vIdx) == vertexMap.end()) vertexMap[vIdx] = {0.0, 0};
            vertexMap[vIdx].first  += (trk->PT * trk->PT); 
            vertexMap[vIdx].second += 1;                   
        }
    }

    bool hasValidPV = false;
    double maxSumPt2 = -1.0;
    int bestPVIndex = -1; 

    for(auto const& [vIdx, data] : vertexMap) {
        if(data.second >= 2 && data.first > maxSumPt2) {
            maxSumPt2 = data.first;
            bestPVIndex = vIdx;
            hasValidPV = true;
        }
    }

    if (applyCuts && !hasValidPV) continue;

    // ==========================================================
    // 2. GATHER CANDIDATE OBJECTS & OVERLAP REMOVAL
    // ==========================================================
    
    // -- Candidates --
    std::vector<Muon*> candMuons;
    for(int i=0; i < branchMuon->GetEntries(); ++i) {
        Muon *mu = (Muon*) branchMuon->At(i);
        if(mu->PT > 7.0 && fabs(mu->Eta) < 2.5) candMuons.push_back(mu);
    }

    std::vector<Electron*> candElectrons;
    for(int i=0; i < branchElectron->GetEntries(); ++i) {
        Electron *ele = (Electron*) branchElectron->At(i);
        double aEta = fabs(ele->Eta);
        if(ele->PT > 7.0 && aEta < 2.5 && !(aEta > 1.37 && aEta < 1.52)) candElectrons.push_back(ele);
    }

    std::vector<Jet*> candJets;
    for(int i=0; i < branchSmallJet->GetEntries(); ++i) {
        Jet *jet = (Jet*) branchSmallJet->At(i);
        if(jet->PT > 30.0) candJets.push_back(jet);
    }

    // -- Overlap Step 1: Ele vs Mu --
    std::vector<Electron*> step1Electrons;
    for(Electron* ele : candElectrons) {
        bool shared = false;
        for(Muon* mu : candMuons) {
            if(CalculateDeltaR(ele->Eta, ele->Phi, mu->Eta, mu->Phi) < 0.05) { shared = true; break; }
        }
        if(!shared) step1Electrons.push_back(ele);
    }

    // -- Overlap Step 2: Jet Cleaning --
    std::vector<Jet*> step2Jets;
    for(Jet* jet : candJets) {
        bool removeJet = false;
        for(Electron* ele : step1Electrons) {
            if(CalculateDeltaR(jet->Eta, jet->Phi, ele->Eta, ele->Phi) < 0.2) { removeJet = true; break; }
        }
        if(!removeJet) {
            for(Muon* mu : candMuons) {
                if(CalculateDeltaR(jet->Eta, jet->Phi, mu->Eta, mu->Phi) < 0.4 && jet->NCharged < 3) {
                    removeJet = true; break;
                }
            }
        }
        if(!removeJet) step2Jets.push_back(jet);
    }

    // -- Overlap Step 3: Lepton Cleaning --
    std::vector<Electron*> finalElectrons;
    for(Electron* ele : step1Electrons) {
        bool overlaps = false;
        for(Jet* jet : step2Jets) {
            if(CalculateDeltaR(ele->Eta, ele->Phi, jet->Eta, jet->Phi) < 0.4) { overlaps = true; break; }
        }
        if(!overlaps) finalElectrons.push_back(ele);
    }

    std::vector<Muon*> finalMuons;
    for(Muon* mu : candMuons) {
        bool overlaps = false;
        for(Jet* jet : step2Jets) {
            if(CalculateDeltaR(mu->Eta, mu->Phi, jet->Eta, jet->Phi) < 0.4) { overlaps = true; break; }
        }
        if(!overlaps) finalMuons.push_back(mu);
    }

    // ==========================================================
    // 3. RECONSTRUCT MET (TREATING MUONS AS INVISIBLE)
    // ==========================================================

    double sumPx = 0.0;
    double sumPy = 0.0;

    // A. Add Selected Electrons (Visible)
    for(Electron* ele : finalElectrons) {
        sumPx += ele->PT * TMath::Cos(ele->Phi);
        sumPy += ele->PT * TMath::Sin(ele->Phi);
    }

    // B. Selected Muons (Invisible)
    // We intentionally SKIP adding muon momentum to sumPx/sumPy here.
    // However, we must still match tracks to them to avoid double counting in the soft term.

    // C. Add Selected Jets (Visible)
    for(Jet* jet : step2Jets) {
        sumPx += jet->PT * TMath::Cos(jet->Phi);
        sumPy += jet->PT * TMath::Sin(jet->Phi);
    }

    // D. Add Unmatched Tracks (Soft Term)
    for(int i = 0; i < nTracks; ++i) {
        Track *trk = (Track*) branchTrack->At(i);
        
        // 1. Must be compatible with Primary Vertex
        if(trk->VertexIndex != bestPVIndex) continue;

        // 2. Check overlap with Selected Objects
        bool matched = false;

        // Match vs Electrons
        for(Electron* ele : finalElectrons) {
            if(CalculateDeltaR(trk->Eta, trk->Phi, ele->Eta, ele->Phi) < 0.05) { matched = true; break; }
        }
        if(matched) continue;

        // Match vs Muons (CRITICAL: Match tracks to muons so we don't add the muon track back!)
        for(Muon* mu : finalMuons) {
            if(CalculateDeltaR(trk->Eta, trk->Phi, mu->Eta, mu->Phi) < 0.05) { matched = true; break; }
        }
        if(matched) continue;

        // Match vs Jets
        for(Jet* jet : step2Jets) {
            if(CalculateDeltaR(trk->Eta, trk->Phi, jet->Eta, jet->Phi) < 0.4) { matched = true; break; }
        }
        if(matched) continue;

        // If not matched to any object (including the invisible muons), add track momentum
        sumPx += trk->PT * TMath::Cos(trk->Phi);
        sumPy += trk->PT * TMath::Sin(trk->Phi);
    }

    // Final Calculation: Negative Vector Sum
    double mPx = -sumPx;
    double mPy = -sumPy;
    double rec_met_val = TMath::Sqrt(mPx*mPx + mPy*mPy);
    double rec_met_phi = TMath::ATan2(mPy, mPx);


    // ==========================================================
    // 4. EVENT SELECTION (Using Rec MET and Cleaned Objects)
    // ==========================================================

    if (applyCuts) {
        Jet *j1 = (step2Jets.size() > 0) ? step2Jets[0] : nullptr;

        // 1. Leading Jet pT > 250
        if (j1 == nullptr || j1->PT <= 250.0) continue;

        // 2. >= 2 Central Jets
        int nCentral = 0;
        for(Jet* jet : step2Jets) {
            if(fabs(jet->Eta) < 2.8) nCentral++;
        }
        if (nCentral < 2) continue;

        // 3. dPhi(Jet, MET) < 2.0 (Using Reconstructed MET)
        bool passDPhi = false;
        for(Jet* jet : step2Jets) {
            double dphi = fabs(TVector2::Phi_mpi_pi(jet->Phi - rec_met_phi));
            if (dphi < 2.0) { passDPhi = true; break; }
        }
        if (!passDPhi) continue;

        // 4. B-Jet Veto
        int nBTags = 0;
        for(Jet* jet : step2Jets) {
            if(jet->BTag == 1) nBTags++;
        }
        if (nBTags >= 2) continue;

        // 5. Tau Veto
        bool failTau = false;
        for(Jet* jet : step2Jets) {
             if (jet->TauTag == 1 && jet->PT > 20.0 && fabs(jet->Eta) < 2.5) {
                 failTau = true; break;
             }
        }
        if (failTau) continue;

        // 6. Lepton Veto
        // Reject events if we found good electrons or muons (even if muons were invisible in MET)
        if (finalElectrons.size() > 0) continue;
        if (finalMuons.size() > 0) continue;
    }

    // ==========================================================
    // 5. VARIABLES & OUTPUT
    // ==========================================================
    
    double ht = 0.0;
    double min_dPhi = 999.0;
    Jet *jetClosest = nullptr;
    Jet *jetFarthest = nullptr;
    double minAbsDPhi = 9999.0;
    double maxAbsDPhi = -1.0;

    for(Jet* jet : step2Jets) {
        ht += jet->PT;
        // Use reconstructed MET phi
        double dphi = fabs(TVector2::Phi_mpi_pi(jet->Phi - rec_met_phi));
        if(dphi < min_dPhi) min_dPhi = dphi;

        if (dphi < minAbsDPhi) { minAbsDPhi = dphi; jetClosest = jet; }
        if (dphi > maxAbsDPhi) { maxAbsDPhi = dphi; jetFarthest = jet; }
    }
    if(step2Jets.empty()) min_dPhi = -1.0;

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

    // FatJets for output
    int nFatJets = branchFatJet->GetEntries();
    Jet *fj1 = (nFatJets > 0) ? (Jet*) branchFatJet->At(0) : nullptr;
    Jet *fj2 = (nFatJets > 1) ? (Jet*) branchFatJet->At(1) : nullptr;
    Jet *cleanJ1 = (step2Jets.size() > 0) ? step2Jets[0] : nullptr;
    Jet *cleanJ2 = (step2Jets.size() > 1) ? step2Jets[1] : nullptr;

    double m_jj = (fj1 && fj2) ? (fj1->P4() + fj2->P4()).M() : -1.0;
    double sm_jj = (cleanJ1 && cleanJ2) ? (cleanJ1->P4() + cleanJ2->P4()).M() : -1.0;

    double tau21_j1 = (fj1 && fj1->Tau[0] > 0) ? fj1->Tau[1] / fj1->Tau[0] : -1.0;
    double tau21_j2 = (fj2 && fj2->Tau[0] > 0) ? fj2->Tau[1] / fj2->Tau[0] : -1.0;
    double tau32_j1 = (fj1 && fj1->Tau[1] > 0) ? fj1->Tau[2] / fj1->Tau[1] : -1.0;
    double tau32_j2 = (fj2 && fj2->Tau[1] > 0) ? fj2->Tau[2] / fj2->Tau[1] : -1.0;

    HepMCEvent *ev = (HepMCEvent*) branchEvent->At(0);
    
    fout << (fj1 ? fj1->PT : -1)  << " " << (fj1 ? fj1->Eta : -1) << " " << (fj1 ? fj1->Phi : -1) << " "
         << (fj2 ? fj2->PT : -1)  << " " << (fj2 ? fj2->Eta : -1) << " " << (fj2 ? fj2->Phi : -1) << " "
         << m_jj << " " << sm_jj << " "
         << tau21_j1 << " " << tau21_j2 << " "
         << tau32_j1 << " " << tau32_j2 << " "
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