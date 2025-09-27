#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

COLUMNS_MC = [
    "pT_j1", "eta_j1", "phi_j1",
    "pT_j2", "eta_j2", "phi_j2",
    "m_jj",
    "tau21_j1", "tau21_j2",
    "tau32_j1", "tau32_j2",
    "met", "phi_met",
    "min_dPhi", "ht",
    "weight"   # include weight column
]
COLUMNS_DATA = [
    "pT_j1", "eta_j1", "phi_j1",
    "pT_j2", "eta_j2", "phi_j2",
    "m_jj",
    "tau21_j1", "tau21_j2",
    "tau32_j1", "tau32_j2",
    "met", "phi_met",
    "min_dPhi", "ht"
]
# Periods
DATA_PERIODS = ["A","B","C","D","E","F","G","I","K","L"]
MC_PERIODS = ["Wjets","Zjets","ttbar","Single_top","Multijet","Diboson"]

# Mapping: var -> (xmin, xmax, nbins, label)
VARIABLES = {
    "pT_j1": (0, 2000, 40, r"$p_{T}^{j1}$ [GeV]"),
    "pT_j2": (0, 1500, 50, r"$p_{T}^{j2}$ [GeV]"),
    "eta_j1": (-3, 3, 50, r"$\eta^{j1}$"),
    "eta_j2": (-3, 3, 50, r"$\eta^{j2}$"),
    "m_jj": (0, 5000, 50, r"$m_{jj}$ [GeV]"),
    "tau21_j1": (0, 1.5, 50, r"$\tau_{21}^{j1}$"),
    "tau21_j2": (0, 1.5, 50, r"$\tau_{21}^{j2}$"),
    "tau32_j1": (0, 1.5, 50, r"$\tau_{32}^{j1}$"),
    "tau32_j2": (0, 1.5, 50, r"$\tau_{32}^{j2}$"),
    "met": (0, 2000, 50, r"$E_{T}^{miss}$ [GeV]"),
    "ht": (0, 5000, 50, r"$H_{T}$ [GeV]"),
    "min_dPhi": (0, 3.2, 50, r"$\min \Delta \phi(jet, MET)$"),
}

# def stream_hist(basepath, period, var, max_events=None):
#     """Compute histogram for one variable in one period."""
#     xmin, xmax, nbins, _ = VARIABLES[var]
#     bins = np.linspace(xmin, xmax, nbins+1)
#     hist = np.zeros(nbins, dtype=float)

#     path = os.path.join(basepath, period, "**/dataset_*.txt")
#     files = glob.glob(path, recursive=True)

#     # Pick columns based on dataset type
#     is_mc = "MC" in basepath
#     columns = COLUMNS_MC if is_mc else COLUMNS_DATA

#     col_index = columns.index(var)
#     weight_index = columns.index("weight") if is_mc else None
#     n_events = 0
#     n_files_used = 0

#     for f in files:
#         if os.path.getsize(f) == 0:
#             continue

#         with open(f) as infile:
#             for i, line in enumerate(infile):
#                 if i == 0 and line.startswith("pT_j1"):  # header
#                     continue

#                 parts = line.strip().split()
#                 if len(parts) != len(columns):
#                     continue
#                 try:
#                     val = float(parts[col_index])
#                     w = float(parts[weight_index]) if is_mc else 1.0
#                 except ValueError:
#                     continue

#                 if xmin <= val < xmax:
#                     h, _ = np.histogram([val], bins=bins, weights=[w])
#                     hist += h
#                     n_events += 1

#                 if max_events and n_events >= max_events:
#                     print(f"[INFO] {period}: used {n_files_used+1}/{len(files)} files, {n_events} valid events")
#                     return hist, bins

#         if n_events > 0:
#             n_files_used += 1

#     print(f"[INFO] {period}: used {n_files_used}/{len(files)} files, {n_events} valid events")
#     return hist, bins

def stream_hist(basepath, period, var, max_events=None):
    """Compute histogram for one variable in one period, with MET < 100 GeV and HT < 200 GeV cuts."""
    xmin, xmax, nbins, _ = VARIABLES[var]
    bins = np.linspace(xmin, xmax, nbins+1)
    hist = np.zeros(nbins, dtype=float)

    path = os.path.join(basepath, period, "**/dataset_*.txt")
    files = glob.glob(path, recursive=True)

    # Pick columns based on dataset type
    is_mc = "MC" in basepath
    columns = COLUMNS_MC if is_mc else COLUMNS_DATA

    col_index = columns.index(var)
    weight_index = columns.index("weight") if is_mc and "weight" in columns else None
    met_index = columns.index("met")
    ht_index = columns.index("ht")

    n_events = 0
    n_files_used = 0

    for f in files:
        if os.path.getsize(f) == 0:
            continue

        with open(f) as infile:
            for i, line in enumerate(infile):
                if i == 0 and line.startswith("pT_j1"):  # header
                    continue

                parts = line.strip().split()
                if len(parts) < len(columns):  # allow for missing weight column
                    continue
                try:
                    val = float(parts[col_index])
                    met_val = float(parts[met_index])
                    ht_val = float(parts[ht_index])
                    w = float(parts[weight_index]) if weight_index is not None else 1.0
                except ValueError:
                    continue

                # Apply event selection: MET < 100, HT < 200
                if not (met_val > 100 and ht_val > 200):
                    continue

                if xmin <= val < xmax:
                    h, _ = np.histogram([val], bins=bins, weights=[w])
                    hist += h
                    n_events += 1

                if max_events and n_events >= max_events:
                    print(f"[INFO] {period}: used {n_files_used+1}/{len(files)} files, {n_events} valid events (after cuts)")
                    return hist, bins

        if n_events > 0:
            n_files_used += 1

    print(f"[INFO] {period}: used {n_files_used}/{len(files)} files, {n_events} valid events (after cuts)")
    return hist, bins


def plot_comparison(vars, data_periods, mc_periods, max_events=None):
    plt.style.use("seaborn-v0_8")
    colors = plt.cm.tab10.colors

    for var in vars:
        xmin, xmax, nbins, xlabel = VARIABLES[var]
        plt.figure(figsize=(8,6))

        # --- Data ---
        hist_data = np.zeros(nbins, dtype=float)
        for period in data_periods:
            h, bins = stream_hist("Dataset_ver2/Data/predataset", period, var, max_events)
            hist_data += h
        centers = 0.5 * (bins[:-1] + bins[1:])

        # --- MC ---
        mc_hists = []
        mc_labels = []
        for i, period in enumerate(mc_periods[::-1]):
            hist_mc, _ = stream_hist("Dataset_ver3/MC/processed", period, var, max_events)
            if hist_mc.sum() == 0:
                continue
            mc_hists.append(hist_mc)
            mc_labels.append(period)

        # # Normalize to shapes
        # if hist_data.sum() > 0:
        #     data_err = np.sqrt(hist_data)/len(hist_data)
        #     data_err = data_err/hist_data.sum()
        #     hist_data = hist_data / hist_data.sum()

            
        # # Normalize MC total to match data.sum()
        # if len(mc_hists) > 0:
        #     stacked = np.sum(mc_hists, axis=0)
        #     scale = hist_data.sum() / stacked.sum() if stacked.sum() > 0 else 1.0
        #     mc_hists = [h * scale for h in mc_hists]

        # # Plot MC stacked
        # if len(mc_hists) > 0:
        #     plt.hist(
        #         [centers] * len(mc_hists), bins=bins, weights=mc_hists,
        #         stacked=True, label=[f"MC {lab}" for lab in mc_labels],
        #         color=colors[:len(mc_hists)], alpha=0.7, edgecolor="black"
        #     )


        # # Plot Data as points
        # if hist_data.sum() > 0:
        #     plt.errorbar(centers, hist_data, yerr= data_err,
        #                  fmt="o", color="black", label="Data")
        # Scale MC total to match *data total*, but don't normalize data
        if len(mc_hists) > 0 and hist_data.sum() > 0:
            stacked = np.sum(mc_hists, axis=0)
            scale = hist_data.sum() / stacked.sum() if stacked.sum() > 0 else 1.0
            mc_hists = [h * scale for h in mc_hists]

        # Plot MC stacked
        if len(mc_hists) > 0:
            plt.hist(
                [centers] * len(mc_hists), bins=bins, weights=mc_hists,
                stacked=True, label=[f"MC {lab}" for lab in mc_labels],
                color=colors[:len(mc_hists)], alpha=0.7, edgecolor="black"
            )

        # Plot Data as points with Poisson errors (raw counts)
        if hist_data.sum() > 0:
            errors = np.sqrt(hist_data)
            plt.errorbar(centers, hist_data, yerr=errors,
                        fmt="o", color="black", label="Data")
   

        plt.xlabel(xlabel)
        plt.ylabel("Normalized events")
        plt.title(f"Data vs MC (stacked, weighted): {xlabel}")
        plt.legend()
        plt.semilogy()
        plt.tight_layout()
        plt.savefig(f"plots/compare_{var}_stacked.png")
        plt.close()
        print(f"[SAVED] compare_{var}_stacked.png")

def main():
    parser = argparse.ArgumentParser(description="Compare Data vs MC distributions")
    parser.add_argument("-var", nargs="+", required=True, help="Variables to plot")
    parser.add_argument("--max-events", type=int, default=None, help="Limit number of events")
    parser.add_argument("-dataperiod", nargs="+", default=DATA_PERIODS, help="Data periods to include")
    parser.add_argument("-mcperiod", nargs="+", default=MC_PERIODS, help="MC periods to include")
    args = parser.parse_args()

    # Expand "all"
    if len(args.var) == 1 and args.var[0].lower() == "all":
        args.var = list(VARIABLES.keys())
    if len(args.dataperiod) == 1 and args.dataperiod[0].lower() == "all":
        args.dataperiod = DATA_PERIODS
    if len(args.mcperiod) == 1 and args.mcperiod[0].lower() == "all":
        args.mcperiod = MC_PERIODS

    for v in args.var:
        if v not in VARIABLES:
            raise ValueError(f"Variable {v} not available. Allowed: {list(VARIABLES.keys())}")

    plot_comparison(args.var, args.dataperiod, args.mcperiod, max_events=args.max_events)

if __name__ == "__main__":
    main()
