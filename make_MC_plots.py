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

PERIODS = ["Wjets","Zjets","ttbar","Single_top","Multijet","Diboson"]

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

def stream_hist(period, var, max_events=None, basepath="Dataset_ver2/MC/processed"):
    """Compute histogram for one variable in one period, streaming from files."""
    xmin, xmax, nbins, _ = VARIABLES[var]
    bins = np.linspace(xmin, xmax, nbins+1)
    hist = np.zeros(nbins, dtype=float)

    path = os.path.join(basepath, period, "**/dataset_*.txt")
    files = glob.glob(path)

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

def plot_variables(vars, periods, max_events=None):
    plt.style.use("seaborn-v0_8")
    colors = plt.cm.tab10.colors

    for var in vars:
        xmin, xmax, nbins, xlabel = VARIABLES[var]
        plt.figure(figsize=(8,6))

        for i, period in enumerate(periods):
            hist, bins = stream_hist(period, var, max_events=max_events)
            if hist.sum() == 0:
                continue

            centers = 0.5 * (bins[:-1] + bins[1:])
            hist = hist / hist.sum()  # normalize

            plt.step(centers, hist, where="mid",
                     color=colors[i % len(colors)], label=f"Period {period}")

        plt.xlabel(xlabel)
        plt.ylabel("Normalized events")
        plt.title(f"Distribution of {xlabel}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/MC/plot_{var}.png")
        plt.close()
        print(f"[SAVED] plot_{var}.png")

def main():
    parser = argparse.ArgumentParser(description="Plot data variables by period (streaming mode)")
    parser.add_argument("-var", nargs="+", required=True, help="Variables to plot")
    parser.add_argument("-period", nargs="+", required=True, help="Data periods to include")
    parser.add_argument("--max-events", type=int, default=None, help="Limit number of events (debugging)")
    args = parser.parse_args()

    # Expand "all" into full lists
    if len(args.var) == 1 and args.var[0].lower() == "all":
        args.var = list(VARIABLES.keys())
    if len(args.period) == 1 and args.period[0].lower() == "all":
        args.period = PERIODS

    for v in args.var:
        if v not in VARIABLES:
            raise ValueError(f"Variable {v} not available. Allowed: {list(VARIABLES.keys())}")
    for p in args.period:
        if p not in PERIODS:
            raise ValueError(f"Period {p} not available. Allowed: {PERIODS}")

    plot_variables(args.var, args.period, max_events=args.max_events)

if __name__ == "__main__":
    main()
