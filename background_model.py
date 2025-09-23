#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

COLUMNS = [
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
    "pT_j1": (0, 2000, 50, r"$p_{T}^{j1}$ [GeV]"),
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

def stream_hist(basepath, period, var, max_events=None):
    """Compute histogram for one variable in one period."""
    xmin, xmax, nbins, _ = VARIABLES[var]
    bins = np.linspace(xmin, xmax, nbins+1)
    hist = np.zeros(nbins, dtype=float)

    path = os.path.join(basepath, period, "**/dataset_*.txt")
    files = glob.glob(path, recursive=True)

    col_index = COLUMNS.index(var)
    n_events = 0

    for f in files:
        with open(f) as infile:
            for line in infile:
                parts = line.strip().split()
                if len(parts) != len(COLUMNS):
                    continue
                try:
                    val = float(parts[col_index])
                except ValueError:
                    continue

                if xmin <= val < xmax:
                    h, _ = np.histogram([val], bins=bins)
                    hist += h
                n_events += 1

                if max_events and n_events >= max_events:
                    return hist, bins
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
        if hist_data.sum() > 0:
            hist_data = hist_data / hist_data.sum()
            plt.step(centers, hist_data, where="mid", color="black", label="Data", linewidth=2)

        # --- MC ---
        for i, period in enumerate(mc_periods):
            hist_mc, _ = stream_hist("Dataset_ver2/MC/processed", period, var, max_events)
            if hist_mc.sum() == 0:
                continue
            hist_mc = hist_mc / hist_mc.sum()
            plt.step(centers, hist_mc, where="mid",
                     color=colors[i % len(colors)], label=f"MC {period}")

        plt.xlabel(xlabel)
        plt.ylabel("Normalized events")
        plt.title(f"Data vs MC: {xlabel}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/compare_{var}.png")
        plt.close()
        print(f"[SAVED] compare_{var}.png")

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
