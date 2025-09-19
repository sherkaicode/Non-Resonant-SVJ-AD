import os
import ROOT
import json
import numpy as np
import sys
import datetime
import argparse
import math
import random

# Suppress ROOT info/warning messages
ROOT.gErrorIgnoreLevel = ROOT.kError

# ------------------ ARGUMENT PARSING ------------------
parser = argparse.ArgumentParser(description="Process ATLAS MC datasets")
parser.add_argument("-process", type=str, required=True,
                    help="Process to run (e.g. Wjets, Zjets, ttbar, Single_top, Multijet, Diboson)")
args = parser.parse_args()
process_to_run = args.process

# ------------------ LOGGING ------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

today = datetime.date.today().strftime("%Y-%m-%d")
existing_logs = [f for f in os.listdir(log_dir) if f.startswith(today)]
run_number = len(existing_logs) + 1
log_file = os.path.join(log_dir, f"{today}_run{run_number}_{process_to_run}.txt")

class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
print(f"Logging to {log_file}\n")

# ------------------ ATLAS MC INFO ------------------
atlas_info = {
    "Wjets": {
        "jsons": [
            "mc20_13TeV_MC_Sh_2211_Wenu_maxHTpTV2_BFilter_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wenu_maxHTpTV2_CFilterBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wenu_maxHTpTV2_CVetoBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wmunu_maxHTpTV2_BFilter_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wmunu_maxHTpTV2_CFilterBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wmunu_maxHTpTV2_CVetoBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wtaunu_L_maxHTpTV2_BFilter_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wtaunu_L_maxHTpTV2_CFilterBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wtaunu_L_maxHTpTV2_CVetoBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wtaunu_H_maxHTpTV2_BFilter_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wtaunu_H_maxHTpTV2_CFilterBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wtaunu_H_maxHTpTV2_CVetoBVeto_file_index.json"
        ],
        "file": "ATLAS_boson.json"
    },
    "Zjets": {
        "jsons": [
            # "mc20_13TeV_MC_Sh_2211_Zmumu_maxHTpTV2_BFilter_file_index.json",
            # "mc20_13TeV_MC_Sh_2211_Zmumu_maxHTpTV2_CFilterBVeto_file_index.json",
            # "mc20_13TeV_MC_Sh_2211_Zmumu_maxHTpTV2_CVetoBVeto_file_index.json",
            # "mc20_13TeV_MC_Sh_2211_Znunu_pTV2_BFilter_file_index.json",
            # "mc20_13TeV_MC_Sh_2211_Znunu_pTV2_CFilterBVeto_file_index.json",
            # "mc20_13TeV_MC_Sh_2211_Znunu_pTV2_CVetoBVeto_file_index.json",
            # "mc20_13TeV_MC_Sh_2214_Ztautau_maxHTpTV2_BFilter_file_index.json", # Not Found
            "mc20_13TeV_MC_Sh_2214_Ztautau_maxHTpTV2_CFilterBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2214_Ztautau_maxHTpTV2_CVetoBVeto_file_index.json"
        ],
        "file": "ATLAS_boson.json"
    },
    "ttbar": {
        "jsons": [
            "mc20_13TeV_MC_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_file_index.json",
            "mc20_13TeV_MC_PhPy8EG_A14_ttbar_hdamp258p75_allhad_file_index.json"
        ],
        "file": "ATLAS_ttbar.json"
    },
    "Single_top": {
        "jsons": [
            "mc20_13TeV_MC_PowhegPythia8EvtGen_A14_singletop_schan_lept_top_file_index.json",
            "mc20_13TeV_MC_PowhegPythia8EvtGen_A14_singletop_schan_lept_antitop_file_index.json",
            "mc20_13TeV_MC_PhPy8EG_A14_tchan_BW50_lept_top_file_index.json",
            "mc20_13TeV_MC_PhPy8EG_A14_tchan_BW50_lept_antitop_file_index.json"
        ],
        "file": "ATLAS_ttbar.json"
    },
    "Multijet": {
        "jsons": [
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ8WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW_file_index.json"
        ],
        "file": "ATLAS_QCD.json"
    },
    "Diboson": {
        "jsons": [
            "mc20_13TeV_MC_Sh_2211_WlvZqq_file_index.json",
            "mc20_13TeV_MC_Sh_2211_WqqZvv_file_index.json",
            "mc20_13TeV_MC_Sh_2211_ZqqZvv_file_index.json",
            "mc20_13TeV_MC_Sh_2211_WlvWqq_file_index.json"
        ],
        "file": "ATLAS_boson.json"
    },
}

# ------------------ LOAD MC METADATA ------------------
def load_metadata(file):
    with open(file, "r") as f:
        return json.load(f)

def get_root_links_from_json(meta, json_name):
    links = []
    for meta_run in meta["metadata"]["_file_indices"]:
        if meta_run["key"] == json_name:
            for root_file in meta_run["files"]:
                links.append(root_file["uri"])
    return links

# ------------------ PREPARE DATASETS ------------------
all_processes = {}
for process, info in atlas_info.items():
    meta = load_metadata(info["file"])
    all_processes[process] = {
        json_name.replace("_file_index.json", ""): get_root_links_from_json(meta, json_name)
        for json_name in info["jsons"]
    }

# ------------------ REDUCE ROOT ------------------
path_reduce_root = "Dataset_ver2/MC/reduce_root"
os.makedirs(path_reduce_root, exist_ok=True)

rel_branches = [
    "AnalysisJetsAuxDyn.eta", "AnalysisJetsAuxDyn.pt", "AnalysisJetsAuxDyn.NNJvtPass",
    "AnalysisJetsAuxDyn.phi", "AnalysisTauJetsAuxDyn.JetDeepSetTight",
    "AnalysisElectronsAuxDyn.DFCommonElectronsLHTight", "AnalysisMuonsAuxDyn.muonType",
    "AnalysisMuonsAuxDyn.quality", "MET_Core_AnalysisMETAuxDyn.mpx",
    "MET_Core_AnalysisMETAuxDyn.mpy", "MET_Core_AnalysisMETAuxDyn.sumet",
    "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pu", "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pc",
    "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pb", "AnalysisJetsAuxDyn.m",
    "AnalysisLargeRJetsAuxDyn.pt", "AnalysisLargeRJetsAuxDyn.eta", "AnalysisLargeRJetsAuxDyn.phi",
    "AnalysisLargeRJetsAuxDyn.m", "AnalysisLargeRJetsAuxDyn.Tau1_wta",
    "AnalysisLargeRJetsAuxDyn.Tau2_wta", "AnalysisLargeRJetsAuxDyn.Tau3_wta"
]

def reduce_root(process, dataset, link, c):
    chain = ROOT.TChain("CollectionTree")
    chain.Add(link)
    df = ROOT.RDataFrame(chain)
    outdir = f"{path_reduce_root}/{process}/{dataset}"
    os.makedirs(outdir, exist_ok=True)
    df.Snapshot("CollectionTree", f"{outdir}/root_{c}.root", rel_branches)

# ------------------ MAKE DATASET ------------------
cut_77 = 0.7
fc = 0.3

def make_dataset(process, dataset, c, reduce_root_file, outdir):
    os.makedirs(outdir, exist_ok=True)
    chain = ROOT.TChain("CollectionTree")
    chain.Add(reduce_root_file)
    df = ROOT.RDataFrame(chain)

    branches = [
        "AnalysisJetsAuxDyn_eta", "AnalysisJetsAuxDyn_pt", "AnalysisJetsAuxDyn_NNJvtPass", "AnalysisJetsAuxDyn_phi",
        "AnalysisTauJetsAuxDyn_JetDeepSetTight", "AnalysisElectronsAuxDyn_DFCommonElectronsLHTight", "AnalysisMuonsAuxDyn_muonType", "AnalysisMuonsAuxDyn_quality",
        "MET_Core_AnalysisMETAuxDyn_mpx", "MET_Core_AnalysisMETAuxDyn_mpy", "MET_Core_AnalysisMETAuxDyn_sumet",
        "BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pu", "BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pc", "BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pb",
        "AnalysisLargeRJetsAuxDyn_pt", "AnalysisLargeRJetsAuxDyn_eta", "AnalysisLargeRJetsAuxDyn_phi",
        "AnalysisLargeRJetsAuxDyn_m", "AnalysisLargeRJetsAuxDyn_Tau1_wta", "AnalysisLargeRJetsAuxDyn_Tau2_wta", "AnalysisLargeRJetsAuxDyn_Tau3_wta"
    ]

    data = df.AsNumpy(branches)

    outfile = os.path.join(outdir, f"dataset_{c}.txt")
    with open(outfile, "w") as fout:
        fout.write("pT_j1 eta_j1 phi_j1 pT_j2 eta_j2 phi_j2 m_jj "
                   "tau21_j1 tau21_j2 tau32_j1 tau32_j2 "
                   "met phi_met min_dPhi ht\n")

        n_events = len(data["AnalysisJetsAuxDyn_pt"])
        for i in range(n_events):
            jet_pt = np.array(data["AnalysisJetsAuxDyn_pt"][i]) / 1000.0
            jet_eta = np.array(data["AnalysisJetsAuxDyn_eta"][i])
            jet_phi = np.array(data["AnalysisJetsAuxDyn_phi"][i])
            rvec_jvt = data["AnalysisJetsAuxDyn_NNJvtPass"][i]
            jvt_pass = np.array([ord(rvec_jvt[j]) for j in range(len(rvec_jvt))], dtype=int)

            pu = np.array(data["BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pu"][i])
            pc = np.array(data["BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pc"][i])
            pb = np.array(data["BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pb"][i])

            rvec_tau = data["AnalysisTauJetsAuxDyn_JetDeepSetTight"][i]
            tau_tight = np.array([ord(rvec_tau[j]) for j in range(len(rvec_tau))], dtype=int)

            rvec_elec = data["AnalysisElectronsAuxDyn_DFCommonElectronsLHTight"][i]
            electron_tight = np.array([ord(rvec_elec[j]) for j in range(len(rvec_elec))], dtype=int)

            rvec_muon = data["AnalysisMuonsAuxDyn_quality"][i]
            muon_Qual = np.array([ord(rvec_muon[j]) for j in range(len(rvec_muon))], dtype=int)
            muon_Type = np.array(data["AnalysisMuonsAuxDyn_muonType"][i])

            mpx = data["MET_Core_AnalysisMETAuxDyn_mpx"][i][0]
            mpy = data["MET_Core_AnalysisMETAuxDyn_mpy"][i][0]
            met = data["MET_Core_AnalysisMETAuxDyn_sumet"][i][0]/1000.0
            phi_met = np.arctan2(mpy, mpx)

            fat_pt = np.array(data["AnalysisLargeRJetsAuxDyn_pt"][i]) / 1000.0
            fat_eta = np.array(data["AnalysisLargeRJetsAuxDyn_eta"][i])
            fat_phi = np.array(data["AnalysisLargeRJetsAuxDyn_phi"][i])
            fat_m   = np.array(data["AnalysisLargeRJetsAuxDyn_m"][i]) / 1000.0
            tau1    = np.array(data["AnalysisLargeRJetsAuxDyn_Tau1_wta"][i])
            tau2    = np.array(data["AnalysisLargeRJetsAuxDyn_Tau2_wta"][i])
            tau3    = np.array(data["AnalysisLargeRJetsAuxDyn_Tau3_wta"][i])

            # --- Cuts ---
            if len(jet_pt) < 2: continue
            if np.sum(np.abs(jet_eta) < 2.8) < 2: continue
            if jet_pt[0] < 250 or jvt_pass[0] != 1: continue
            if jet_pt[1] < 30 or jvt_pass[1] != 1: continue
            dphis = [abs(ROOT.TVector2.Phi_mpi_pi(jphi - phi_met)) for jphi in jet_phi]
            if np.sum(np.array(dphis) < 2.0) <= 1: continue
            if np.sum(np.log(pb/(fc*pc + (1-fc)*pu)) > cut_77) >= 2: continue
            if np.sum(tau_tight == 1) > 0: continue
            if np.sum(electron_tight == 1) > 0: continue
            if np.sum(muon_Type == 0) > 0 and np.sum((muon_Qual == 8) | (muon_Qual == 9)) > 0: continue
            if len(fat_pt) < 2: continue

            j1 = ROOT.TLorentzVector(); j1.SetPtEtaPhiM(fat_pt[0], fat_eta[0], fat_phi[0], fat_m[0])
            j2 = ROOT.TLorentzVector(); j2.SetPtEtaPhiM(fat_pt[1], fat_eta[1], fat_phi[1], fat_m[1])
            m_jj = (j1+j2).M()

            tau21_j1 = tau2[0]/tau1[0] if tau1[0] > 0 else -1
            tau21_j2 = tau2[1]/tau1[1] if tau1[1] > 0 else -1
            tau32_j1 = tau3[0]/tau2[0] if tau2[0] > 0 else -1
            tau32_j2 = tau3[1]/tau2[1] if tau2[1] > 0 else -1
            min_dPhi = np.min(dphis)
            ht = np.sum(jet_pt)

            row = [
                fat_pt[0], fat_eta[0], fat_phi[0],
                fat_pt[1], fat_eta[1], fat_phi[1],
                m_jj,
                tau21_j1, tau21_j2,
                tau32_j1, tau32_j2,
                met, phi_met, min_dPhi, ht
            ]
            fout.write(" ".join(map(str, row)) + "\n")

# ------------------ MAIN LOOP ------------------
if process_to_run not in all_processes:
    print(f"ERROR: Process '{process_to_run}' not found! Available: {list(all_processes.keys())}")
    sys.exit(1)

datasets = all_processes[process_to_run]
print(f"\n=== Processing Process {process_to_run} ===")

for dataset_name, links in datasets.items():
    print(f"\n--- Dataset {dataset_name}: {len(links)} files ---")

    n_select = math.ceil(len(links) * 0.6)
    selected_links = random.sample(links, n_select)
    print(f"Selected {n_select} files (60%)")

    reduce_outdir = f"{path_reduce_root}/{process_to_run}/{dataset_name}"
    dataset_outdir = f"Dataset_ver2/MC/processed/{process_to_run}/{dataset_name}"
    os.makedirs(dataset_outdir, exist_ok=True)

    for c, link in enumerate(selected_links):
        print(f"[{c+1}/{n_select}] Processing {link}")
        reduce_root(process_to_run, dataset_name, link, c)
        reduced_file = f"{reduce_outdir}/root_{c}.root"
        make_dataset(process_to_run, dataset_name, c, reduced_file, dataset_outdir)

print(f"\n✅ Finished processing {process_to_run}. Results saved in Dataset_ver2/MC/processed/{process_to_run}/")
