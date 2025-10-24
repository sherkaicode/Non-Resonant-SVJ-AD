/*
Simple macro showing how to access branches from the delphes output root file,
loop over events, and plot simple quantities such as the jet pt.

root -l examples/Analyzer.C'("delphes_output.root", "output.txt")'
*/

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#else
class ExRootTreeReader;
class ExRootResult;
#endif

//------------------------------------------------------------------------------


void AnalyseEvents(ExRootTreeReader *treeReader,  const char *outputFile_part)
{

  // Get pointers to branches used in this analysis
  // --- Branch connections ---
  TClonesArray *branchEvent     = treeReader->UseBranch("Event");
  TClonesArray *branchFatJet    = treeReader->UseBranch("FatJet");
  TClonesArray *branchSmallJet  = treeReader->UseBranch("SmallJet");
  TClonesArray *branchScalarHT  = treeReader->UseBranch("ScalarHT");
  TClonesArray *branchMET       = treeReader->UseBranch("MissingET");
  TClonesArray *branchElectron  = treeReader->UseBranch("Electron");
  TClonesArray *branchMuon      = treeReader->UseBranch("Muon");

  Long64_t allEntries = treeReader->GetEntries();
  
  ofstream myfile_part;

  cout << "** Chain contains " << allEntries << " events" << endl;

  myfile_part.open (outputFile_part);
  
  myfile_part << "pT_j1"<< " " << "eta_j1" << " " << "phi_j1" << " " << "pT_j2" << " " << "eta_j2" << " " << "phi_j2" << " " << "m_jj" << " " << "tau21_j1" << " " << "tau21_j2" << " " << "tau32_j1" << " " << "tau32_j2" << " " << "met" << " " << "phi_met" << " " << "min_dPhi" << " " << "ht" << " " << std::endl;

  int nEvent0fatjets = 0;
  int nEvent1fatjets = 0;
  int nEvent2fatjets = 0;
  int nEvent0smalljets = 0;
  int nEvent1smalljets = 0;
  int nEvent2smalljets = 0;

  int nEventSelected = 0;

  // Loop over all events
  for(Int_t entry = 0; entry < allEntries; ++entry)
  {
    // Load selected branches with data from specified event
    treeReader->ReadEntry(entry);
    
    if(entry%1000 == 0) cout << "Event number: "<< entry <<endl;
    // --- Require at least two FatJets ---
    if (branchFatJet->GetEntries() < 2) continue;
    // --- Require at least two SmallJets ---
    if (branchSmallJet->GetEntries() < 2) continue;
    if (branchElectron->GetEntries() > 0) continue;
    if (branchMuon->GetEntries() > 0) continue;
    Float_t fat_pT_j1, fat_pT_j2, fat_eta_j1, fat_eta_j2, fat_phi_j1, fat_phi_j2;
    Float_t small_pT_j1, small_pT_j2, small_eta_j1, small_eta_j2, small_phi_j1, small_phi_j2;
    Float_t tau21_j1, tau21_j2, tau32_j1, tau32_j2;
    Float_t m_jj, met, phi_met, ht;

    Jet *fatjet1, *fatjet2;
    Jet *smalljet1, *smalljet2;
    MissingET *MET1;
    ScalarHT *HT1;


    // if(branchFatJet->GetEntries() == 0) nEvent0fatjets++;
    // if(branchFatJet->GetEntries() == 1) nEvent1fatjets++;
    // if(branchFatJet->GetEntries() == 2) nEvent2fatjets++;
    // if(branchSmallJet->GetEntries() == 0) nEvent0smalljets++;
    // if(branchSmallJet->GetEntries() == 1) nEvent1smalljets++;
    // if(branchSmallJet->GetEntries() == 2) nEvent2smalljets++;

    // --- Get FatJets ---
    fatjet1 = (Jet*) branchFatJet->At(0);
    fatjet2 = (Jet*) branchFatJet->At(1);
    // --- Get SmallJets ---
    smalljet1 = (Jet*) branchSmallJet->At(0);
    smalljet2 = (Jet*) branchSmallJet->At(1);

    // --- HT and MET ---
    HT1 = (ScalarHT*) branchScalarHT->At(0);
    MET1 = (MissingET*) branchMET->At(0);

    // --- Assign quantities ---
    fat_pT_j1 = fatjet1->PT;
    fat_eta_j1 = fatjet1->Eta;
    fat_phi_j1 = fatjet1->Phi;

    fat_pT_j2 = fatjet2->PT;
    fat_eta_j2 = fatjet2->Eta;
    fat_phi_j2 = fatjet2->Phi;

    small_pT_j1 = smalljet1->PT;
    small_eta_j1 = smalljet1->Eta;
    small_phi_j1 = smalljet1->Phi;

    small_pT_j2 = smalljet2->PT;
    small_eta_j2 = smalljet2->Eta;
    small_phi_j2 = smalljet2->Phi;

    // --- Derived quantities ---
    tau21_j1 = (fatjet1->Tau[0] > 0) ? fatjet1->Tau[1] / fatjet1->Tau[0] : -1;
    tau21_j2 = (fatjet2->Tau[0] > 0) ? fatjet2->Tau[1] / fatjet2->Tau[0] : -1;
    tau32_j1 = (fatjet1->Tau[1] > 0) ? fatjet1->Tau[2] / fatjet1->Tau[1] : -1;
    tau32_j2 = (fatjet2->Tau[1] > 0) ? fatjet2->Tau[2] / fatjet2->Tau[1] : -1;

    met = MET1->MET;
    phi_met = MET1->Phi;
    ht = HT1->HT;

    m_jj = ((fatjet1->P4()) + (fatjet2->P4())).M();

    // if (s)mall_pT_j1 < 250.0 | small_pT_j2 < 30.0; continue;
    // 2. Require ≥ 2 central jets |η| < 2.8
    int nCentral = 0;
    for (int i = 0; i < branchSmallJet->GetEntries(); i++)
    {
      Jet *sj = (Jet*) branchSmallJet->At(i);
      if (fabs(sj->Eta) < 2.8) nCentral++;
    }
    if (nCentral < 2) continue;

    // 3. pT cuts on leading and subleading small jets
    if (small_pT_j1 < 250.0) continue;
    if (small_pT_j2 < 30.0) continue;


    // 4. Δφ(jet, MET) requirement
    Float_t dPhi_j1 = fabs(TVector2::Phi_mpi_pi(small_phi_j1 - phi_met));
    Float_t dPhi_j2 = fabs(TVector2::Phi_mpi_pi(small_phi_j2 - phi_met));
    int nLargeDPhi = 0;
    if (dPhi_j1 > 2.0) nLargeDPhi++;
    if (dPhi_j2 > 2.0) nLargeDPhi++;
    if (nLargeDPhi <= 1) continue;

    // 5. b-tag veto: fewer than 2 b-tagged small jets
    int nBtags = 0;
    for (int i = 0; i < branchSmallJet->GetEntries(); i++)
    {
      Jet *sj = (Jet*) branchSmallJet->At(i);
      if (sj->BTag == 1) nBtags++;
    }
    if (nBtags >= 2) continue;

    // 6. tau veto using SmallJet.TauTag
    int nTauTag = 0;
    for (int i = 0; i < branchSmallJet->GetEntries(); i++)
    {
      Jet *sj = (Jet*) branchSmallJet->At(i);
      if (sj->TauTag == 1) nTauTag++;
    }
    if (nTauTag > 0) continue;

    // --- Min Δφ between MET and FatJets ---
    Float_t dPhi_fj1 = fabs(TVector2::Phi_mpi_pi(fat_phi_j1 - phi_met));
    Float_t dPhi_fj2 = fabs(TVector2::Phi_mpi_pi(fat_phi_j2 - phi_met));
    Float_t min_dPhi = std::min(dPhi_fj1, dPhi_fj2);

    myfile_part << fat_pT_j1 << " " << fat_eta_j1 << " " << fat_phi_j1 << " "
                << fat_pT_j2 << " " << fat_eta_j2 << " " << fat_phi_j2 << " "
                << m_jj << " "
                << tau21_j1 << " " << tau21_j2 << " "
                << tau32_j1 << " " << tau32_j2 << " "
                << met << " " << phi_met << " "
                << min_dPhi << " " << ht << std::endl;
    nEventSelected++;
    
  }
  cout << "** Selected events: " << nEventSelected
       << " / " << allEntries << " processed." << endl;

  myfile_part.close();

  // cout << "nEvent0jets: " << nEvent0jets << std::endl;
  // cout << "nEvent1jets: " << nEvent1jets << std::endl;
  // cout << "nEvent2jets: " << nEvent2jets << std::endl;

}

//------------------------------------------------------------------------------

void Analyzer(const char *inputFile, const char *outputFile_part)
{
  gSystem->Load("libDelphes");

  TChain *chain = new TChain("Delphes");
  chain->Add(inputFile);

  ExRootTreeReader *treeReader = new ExRootTreeReader(chain);

  AnalyseEvents(treeReader, outputFile_part);

  cout << "** Exiting..." << endl;

  delete treeReader;
  delete chain;
  exit(0);
}

//------------------------------------------------------------------------------

