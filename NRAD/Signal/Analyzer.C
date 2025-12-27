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
  TClonesArray *branchFatJet    = treeReader->UseBranch("FatJet");
  TClonesArray *branchSmallJet  = treeReader->UseBranch("Jet");
  TClonesArray *branchScalarHT  = treeReader->UseBranch("ScalarHT");
  TClonesArray *branchMET       = treeReader->UseBranch("MissingET");
  TClonesArray *branchElectron  = treeReader->UseBranch("Electron");
  TClonesArray *branchMuon      = treeReader->UseBranch("Muon");
  TClonesArray* branchEvent = treeReader->UseBranch("Event");

  Long64_t allEntries = treeReader->GetEntries();
  ofstream myfile_part(outputFile_part);

  cout << "** Chain contains " << allEntries << " events" << endl;

  myfile_part << "pT_j1 eta_j1 phi_j1 pT_j2 eta_j2 phi_j2 m_jj "
            << "tau21_j1 tau21_j2 tau32_j1 tau32_j2 "
            << "met phi_met min_dPhi ht "
            << "weight cross_section"
            << endl;


  int nEventSelected = 0;
  bool cut = false;

  for (Int_t entry = 0; entry < allEntries; ++entry)
  {
    treeReader->ReadEntry(entry);
    if (entry % 1000 == 0) cout << "Event: " << entry << endl;

    Jet *fat1 = branchFatJet->GetEntries() > 0 ? (Jet*) branchFatJet->At(0) : nullptr;
    Jet *fat2 = branchFatJet->GetEntries() > 1 ? (Jet*) branchFatJet->At(1) : nullptr;

    Jet *j1 = branchSmallJet->GetEntries() > 0 ? (Jet*) branchSmallJet->At(0) : nullptr;
    Jet *j2 = branchSmallJet->GetEntries() > 1 ? (Jet*) branchSmallJet->At(1) : nullptr;

    MissingET *MET1 = (MissingET*) branchMET->At(0);
    Float_t met = MET1->MET;
    Float_t phi_met = MET1->Phi;

    if (cut == true) {
    // ------------------------------
    // 0. Fat jet requirement (≥2)
    // ------------------------------
      if (branchFatJet->GetEntries() < 2) continue;

      // ------------------------------
      // 1. Small-R jets (≥2)
      // ------------------------------
      if (branchSmallJet->GetEntries() < 2) continue;

      // ------------------------------
      // 2. Lepton veto
      // Data/MC uses *tight* leptons; Delphes lacks this → best approximation
      // ------------------------------
      if (branchElectron->GetEntries() > 0) continue;
      if (branchMuon->GetEntries() > 0) continue;

      // --------------------------------------------------------
      // 3. Require ≥2 central jets (|eta|<2.8)
      // --------------------------------------------------------
      int nCentral = 0;
      for (int i = 0; i < branchSmallJet->GetEntries(); i++)
      {
        Jet *sj = (Jet*) branchSmallJet->At(i);
        if (fabs(sj->Eta) < 2.8) nCentral++;
      }
      if (nCentral < 2) continue;

      // --------------------------------------------------------
      // 4. pT cuts on leading small jets
      // --------------------------------------------------------
      if (j1->PT < 250.0) continue;
      if (j2->PT < 30.0) continue;

      // --------------------------------------------------------
      // 5. Δφ(jet, MET) requirement (MATCHED WITH DATA/MC)
      //
      // Data/MC: require ≥2 jets with Δφ < 2.0
      //
      // --------------------------------------------------------
      int nDPhiSmall = 0;
      for (int i = 0; i < branchSmallJet->GetEntries(); i++)
      {
        Jet *sj = (Jet*) branchSmallJet->At(i);
        float dphi = fabs(TVector2::Phi_mpi_pi(sj->Phi - phi_met));
        if (dphi < 2.0) nDPhiSmall++;
      }
      if (nDPhiSmall <= 1) continue;

      // --------------------------------------------------------
      // 6. b-tag veto: fewer than 2 b-tagged jets
      // (Delphes BTag approximates DL1dv01 truth-level)
      // --------------------------------------------------------
      int nBTags = 0;
      for (int i = 0; i < branchSmallJet->GetEntries(); i++)
      {
        Jet *sj = (Jet*) branchSmallJet->At(i);
        if (sj->BTag == 1) nBTags++;
      }
      if (nBTags >= 2) continue;

      // --------------------------------------------------------
      // 7. tau veto (equivalent intention)
      // --------------------------------------------------------
      int nTau = 0;
      for (int i = 0; i < branchSmallJet->GetEntries(); i++)
      {
        Jet *sj = (Jet*) branchSmallJet->At(i);
        if (sj->TauTag == 1) nTau++;
      }
      if (nTau > 0) continue;
    }

    // --------------------------------------------------------
    // 8. Fat jet-derived variables
    // --------------------------------------------------------
    float tau21_1 = (fat1 && fat1->Tau[0] > 0) ? fat1->Tau[1]/fat1->Tau[0] : -1;
    float tau21_2 = (fat2 && fat2->Tau[0] > 0) ? fat2->Tau[1]/fat2->Tau[0] : -1;

    float tau32_1 = (fat1 && fat1->Tau[1] > 0) ? fat1->Tau[2]/fat1->Tau[1] : -1;
    float tau32_2 = (fat2 && fat2->Tau[1] > 0) ? fat2->Tau[2]/fat2->Tau[1] : -1;

    // safe m_jj: only compute if both fat1 and fat2 exist
    float m_jj = (fat1 && fat2) ? (fat1->P4() + fat2->P4()).M() : -1;


    // --------------------------------------------------------
    // 9. Minimum Δφ between MET and fat jets
    // --------------------------------------------------------
    float dphi_f1 = (fat1) ? fabs(TVector2::Phi_mpi_pi(fat1->Phi - phi_met)) : 1e6;
    float dphi_f2 = (fat2) ? fabs(TVector2::Phi_mpi_pi(fat2->Phi - phi_met)) : 1e6;

    // Use a large default for min_dphi if any fat jet is missing
    float min_dphi = std::min(dphi_f1, dphi_f2);


    // --------------------------------------------------------
    // 10. HT (match data/MC definition: sum of small jet pT)
    // --------------------------------------------------------
    float ht = 0;
    for (int i = 0; i < branchSmallJet->GetEntries(); i++)
    {
      Jet *sj = (Jet*) branchSmallJet->At(i);
      ht += sj->PT;
    }
    HepMCEvent *ev = (HepMCEvent*) branchEvent->At(0);

    float weight = ev->Weight;
    float xsec   = ev->CrossSection;

    // --------------------------------------------------------
    // Save selected event
    // --------------------------------------------------------
    myfile_part 
    << (fat1 ? fat1->PT  : -1) << " "
    << (fat1 ? fat1->Eta : -1) << " "
    << (fat1 ? fat1->Phi : -1) << " "
    << (fat2 ? fat2->PT  : -1) << " "
    << (fat2 ? fat2->Eta : -1) << " "
    << (fat2 ? fat2->Phi : -1) << " "
    << m_jj << " "
    << tau21_1 << " " << tau21_2 << " "
    << tau32_1 << " " << tau32_2 << " "
    << met << " " << phi_met << " "
    << min_dphi << " " << ht << " "
    << weight << " " << xsec
    << endl;



    nEventSelected++;
  }

  cout << "** Selected events: " << nEventSelected
       << " / " << allEntries << endl;
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

