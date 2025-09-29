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
    "min_dPhi", "ht", 
    "weight"
]
MC_COLORS = {
    "Wjets": "cyan",        # light cyan
    "Zjets": "green",       # green
    "ttbar": "purple",       # purple
    "Single_top": "navy",    # dark blue
    "Diboson": "gold",       # yellow
    "Multijet": "turquoise"  # dark cyan / turquoise
}

PERIODS = ["Wjets","Zjets","ttbar","Single_top","Multijet","Diboson"]

# Mapping: var -> (xmin, xmax, nbins, label)
VARIABLES = {
    "pT_j1": (0, 2000, 50, r"$p_{T}^{j1}$ [GeV]", True),
    "pT_j2": (0, 1500, 50, r"$p_{T}^{j2}$ [GeV]", True),
    "eta_j1": (-3, 3, 50, r"$\eta^{j1}$", False),
    "eta_j2": (-3, 3, 50, r"$\eta^{j2}$", False),
    "m_jj": (0, 5000, 50, r"$m_{jj}$ [GeV]", True),
    "tau21_j1": (0, 1.5, 50, r"$\tau_{21}^{j1}$", False),
    "tau21_j2": (0, 1.5, 50, r"$\tau_{21}^{j2}$", False),
    "tau32_j1": (0, 1.5, 50, r"$\tau_{32}^{j1}$", False),
    "tau32_j2": (0, 1.5, 50, r"$\tau_{32}^{j2}$", False),
    "met": (0, 2000, 50, r"$E_{T}^{miss}$ [GeV]", True),
    "ht": (0, 5000, 50, r"$H_{T}$ [GeV]", True),
    "min_dPhi": (0, 3.2, 50, r"$\min \Delta \phi(jet, MET)$", False),
}

def stream_hist(period, var, max_events=None, basepath="Dataset_ver3/MC/processed"):
    xmin, xmax, nbins, _, _ = VARIABLES[var]
    bins = np.linspace(xmin, xmax, nbins+1)
    hist = np.zeros(nbins, dtype=float)

    path = os.path.join(basepath, period, "**/dataset_*.txt")
    files = glob.glob(path)

    col_index = COLUMNS.index(var)
    weight_index = COLUMNS.index("weight")
    n_events = 0

    for f in files:
        with open(f) as infile:
            for line in infile:
                parts = line.strip().split()
                if len(parts) != len(COLUMNS):
                    continue
                try:
                    val = float(parts[col_index])
                    w   = float(parts[weight_index])
                except ValueError:
                    continue

                if xmin <= val < xmax:
                    h, _ = np.histogram([val], bins=bins, weights=[w])
                    hist += h
                n_events += 1

                if max_events and n_events >= max_events:
                    return hist, bins

    return hist, bins


def plot_variables(vars, periods, max_events=None, basepath="Dataset_ver3/MC/processed"):
    plt.style.use("seaborn-v0_8")

    for var in vars:
        xmin, xmax, nbins, xlabel, _ = VARIABLES[var]
        bins = np.linspace(xmin, xmax, nbins+1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        all_hists, labels, colors_used = [], [], []

        for period in periods:
            hist, _ = stream_hist(period, var, max_events=max_events)
            if hist.sum() == 0:
                continue
            all_hists.append(hist)
            labels.append(period)
            colors_used.append(MC_COLORS.get(period, "gray"))

        if not all_hists:
            print(f"[SKIPPED] {var} (no events found)")
            continue

        all_hists = np.array(all_hists)

        fig, ax = plt.subplots(figsize=(10, 6))
        bottom = np.zeros_like(bin_centers)

        for i in range(all_hists.shape[0]):
            ax.bar(
                bin_centers,
                all_hists[i],
                width=(bins[1] - bins[0]),
                bottom=bottom,
                color=colors_used[i],
                label=labels[i],
                linewidth=0.6,
                edgecolor="black",
                alpha=0.8
            )
            bottom += all_hists[i]

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Event Count")
        ax.set_title(f"Distribution of {xlabel}")
        ax.legend(loc="upper right", fontsize="small", ncol=2)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
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
