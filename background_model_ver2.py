#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.gridspec import GridSpec

# --- Column definitions ---
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

# --- Periods ---
DATA_PERIODS = ["A","B","C","D","E","F","G","I","K","L"]
MC_PERIODS = ["Wjets","Zjets","ttbar","Single_top","Multijet","Diboson"]
# MC_COLORS = {
#     "Wjets": "cyan",          # light cyan
#     "Zjets": "green",         # green
#     "ttbar": "purple",         # purple
#     "Single_top": "navy",      # dark blue
#     "Diboson": "gold",         # yellow
#     "Multijet": "turquoise",   # dark cyan / turquoise
# }


# --- Variables ---
VARIABLES = {
    "pT_j1": (0, 600, 20, r"$p_{T}^{j1}$ [GeV]", True),
    "pT_j2": (0, 600, 20, r"$p_{T}^{j2}$ [GeV]", True),
    "eta_j1": (-3, 3, 15, r"$\eta^{j1}$", False),
    "eta_j2": (-3, 3, 15, r"$\eta^{j2}$", False),
    "m_jj": (0, 5000, 20, r"$m_{jj}$ [GeV]", True),
    "tau21_j1": (0, 1.5, 15, r"$\tau_{21}^{j1}$", False),
    "tau21_j2": (0, 1.5, 15, r"$\tau_{21}^{j2}$", False),
    "tau32_j1": (0, 1.5, 15, r"$\tau_{32}^{j1}$", False),
    "tau32_j2": (0, 1.5, 15, r"$\tau_{32}^{j2}$", False),
    "met": (0, 700, 20, r"$E_{T}^{miss}$ [GeV]", True),
    "ht": (0, 700, 20, r"$H_{T}$ [GeV]", True),
    "min_dPhi": (0, 3.0, 15, r"$\min \Delta \phi(jet, MET)$", False),
}

# --- Stream histogram ---
def stream_hist(basepath, period, var, max_events=None):
    """Compute histogram for one variable in one period, with optional max events and MET/HT cuts."""
    xmin, xmax, nbins, _, _ = VARIABLES[var]
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
                # try:
                val = float(parts[col_index])
                met_val = float(parts[met_index])
                ht_val = float(parts[ht_index])
                w = float(parts[weight_index]) if weight_index is not None else 1.0
                # except ValueError:
                #     continue

                in_sr = (ht_val < 600 and met_val < 600)
                if not in_sr:
                    continue
                # Example cut: keep all events (change logic if needed)
                # if (met_val < 600 and ht_val < 600):
                #     continue

                if xmin <= val < xmax:
                    h, _ = np.histogram([val], bins=bins, weights=[w])
                    hist += h
                    n_events += 1

                if max_events and n_events >= max_events:
                    print(f"[INFO] {period}: used {n_files_used+1}/{len(files)} files, {n_events} valid events")
                    return hist, bins

        if n_events > 0:
            n_files_used += 1

    print(f"[INFO] {period}: used {n_files_used}/{len(files)} files, {n_events} valid events")
    return hist, bins

# # --- Plot comparison ---
# def plot_comparison(vars, data_periods, mc_periods, max_data_events=None, max_mc_events=None):
#     plt.style.use("seaborn-v0_8")
#     colors = plt.cm.tab10.colors

#     for var in vars:
#         xmin, xmax, nbins, xlabel, log = VARIABLES[var]
#         plt.figure(figsize=(8,6))

#         # --- Data ---
#         hist_data = np.zeros(nbins, dtype=float)
#         for period in data_periods:
#             h, bins = stream_hist("Dataset_ver2/Data/predataset", period, var, max_events=max_data_events)
#             hist_data += h
#         centers = 0.5 * (bins[:-1] + bins[1:])

#         # --- MC ---
#         mc_hists = []
#         mc_labels = []
#         for i, period in enumerate(mc_periods[::-1]):
#             hist_mc, _ = stream_hist("Dataset_ver3/MC/processed", period, var, max_events=max_mc_events)
#             if hist_mc.sum() == 0:
#                 continue
#             mc_hists.append(hist_mc)
#             mc_labels.append(period)

#         # Scale MC total to match data (no normalization of data)
#         if len(mc_hists) > 0 and hist_data.sum() > 0:
#             stacked = np.sum(mc_hists, axis=0)
#             scale = hist_data.sum() / stacked.sum() if stacked.sum() > 0 else 1.0
#             mc_hists = [h * scale for h in mc_hists]


#         if len(mc_hists) > 0:
#             width = bins[1] - bins[0]
#             bottom = np.zeros_like(centers)

#             for i, h in enumerate(mc_hists):
#                 plt.bar(
#                     centers, h,
#                     width=width,
#                     bottom=bottom,
#                     color=MC_COLORS,
#                     edgecolor="black", linewidth=0.5,
#                     label=f"MC {mc_labels[i]}",
#                     align="center"
#                 )
#                 bottom += h



#         # Plot Data with Poisson errors
#         if hist_data.sum() > 0:
#             errors = np.sqrt(hist_data)
#             plt.errorbar(centers, hist_data, yerr=errors,
#                          fmt="o", color="black", label="Data")

#         plt.xlabel(xlabel)
#         plt.ylabel("Events")
#         plt.title(f"Data vs MC (stacked, weighted): {xlabel}")
#         plt.legend()
#         plt.semilogy(log)
#         plt.tight_layout()
#         plt.savefig(f"plots/compare_{var}_stacked.png")
#         plt.close()
#         print(f"[SAVED] compare_{var}_stacked.png")

MC_COLORS = {
    "Wjets": "cyan",        # light cyan
    "Zjets": "green",       # green
    "ttbar": "purple",       # purple
    "Single_top": "navy",    # dark blue
    "Diboson": "gold",       # yellow
    "Multijet": "turquoise"  # dark cyan / turquoise
}

def plot_comparison(vars, data_periods, mc_periods, max_data_events=None, max_mc_events=None):
    plt.style.use("seaborn-v0_8")

    for var in vars:
        xmin, xmax, nbins, xlabel, log = VARIABLES[var]

        # Set up figure with ratio subplot
        fig = plt.figure(figsize=(8,7))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

        # --- Data ---
        hist_data = np.zeros(nbins, dtype=float)
        for period in data_periods:
            h, bins = stream_hist("Dataset_ver2/Data/predataset", period, var, max_events=max_data_events)
            hist_data += h
        centers = 0.5 * (bins[:-1] + bins[1:])

        # --- MC ---
        mc_hists = []
        mc_labels = []
        for period in mc_periods[::-1]:
            hist_mc, _ = stream_hist("Dataset_ver3/MC/processed", period, var, max_events=max_mc_events)
            if hist_mc.sum() == 0:
                continue
            mc_hists.append(hist_mc)
            mc_labels.append(period)

        # Scale MC total to match data
        if len(mc_hists) > 0 and hist_data.sum() > 0:
            stacked = np.sum(mc_hists, axis=0)
            scale = hist_data.sum() / stacked.sum() if stacked.sum() > 0 else 1.0
            mc_hists = [h * scale for h in mc_hists]
            stacked = np.sum(mc_hists, axis=0)
        else:
            stacked = np.zeros_like(hist_data)

        # --- Draw stacked MC ---
        if len(mc_hists) > 0:
            width = bins[1] - bins[0]
            bottom = np.zeros_like(centers)

            for i, h in enumerate(mc_hists):
                label = mc_labels[i]
                color = MC_COLORS.get(label, "gray")  # fallback color

                ax_main.bar(
                    centers, h,
                    width=width,
                    bottom=bottom,
                    color=color,
                    edgecolor="black", linewidth=0.5,
                    label=label,
                    align="center"
                )
                bottom += h

        # --- Data with Poisson errors ---
        if hist_data.sum() > 0:
            errors = np.sqrt(hist_data)
            ax_main.errorbar(
                centers, hist_data, yerr=errors,
                fmt="o", color="black", label="Data"
            )

        # --- Ratio plot (Data / Bkg) ---
        ratio = np.divide(hist_data, stacked, out=np.zeros_like(hist_data), where=stacked > 0)
        ratio_err = np.divide(np.sqrt(hist_data), stacked, out=np.zeros_like(hist_data), where=stacked > 0)

        ax_ratio.errorbar(
            centers, ratio, yerr=ratio_err,
            fmt="o", color="black"
        )

        # Uncertainty band (MC stat only here)
        mc_err = np.sqrt(stacked)  # simple MC stat unc
        mc_relerr = np.divide(mc_err, stacked, out=np.zeros_like(mc_err), where=stacked > 0)
        ax_ratio.fill_between(
            centers, 1 - mc_relerr, 1 + mc_relerr,
            step="mid", color="gray", alpha=0.4, hatch="//", edgecolor="gray", linewidth=0.0
        )

        # --- Formatting ---
        ax_main.set_ylabel("Events")
        if log:
            ax_main.set_yscale("log")
        ax_main.legend()
        ax_main.set_title(f"Data vs MC (stacked, weighted): {xlabel}")

        ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax_ratio.set_ylabel("Data/Bkg.")
        ax_ratio.set_xlabel(xlabel)
        ax_ratio.set_ylim(0.5, 1.5)

        # Remove x labels from top plot
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Save
        plt.tight_layout()
        plt.savefig(f"plots/compare_{var}_stacked.png")
        plt.close()
        print(f"[SAVED] compare_{var}_stacked.png")

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Compare Data vs MC distributions")
    parser.add_argument("-var", nargs="+", required=True, help="Variables to plot")
    parser.add_argument("-dataperiod", nargs="+", default=DATA_PERIODS, help="Data periods to include")
    parser.add_argument("-mcperiod", nargs="+", default=MC_PERIODS, help="MC periods to include")
    parser.add_argument("-maxDataevents", default="all", help="Max events for Data (int or 'all')")
    parser.add_argument("-maxMCevents", default="all", help="Max events for MC (int or 'all')")
    args = parser.parse_args()

    # Expand "all"
    if len(args.var) == 1 and args.var[0].lower() == "all":
        args.var = list(VARIABLES.keys())
    if len(args.dataperiod) == 1 and args.dataperiod[0].lower() == "all":
        args.dataperiod = DATA_PERIODS
    if len(args.mcperiod) == 1 and args.mcperiod[0].lower() == "all":
        args.mcperiod = MC_PERIODS

    # Parse max events
    max_data_events = None if str(args.maxDataevents).lower() == "all" else int(args.maxDataevents)
    max_mc_events = None if str(args.maxMCevents).lower() == "all" else int(args.maxMCevents)

    for v in args.var:
        if v not in VARIABLES:
            raise ValueError(f"Variable {v} not available. Allowed: {list(VARIABLES.keys())}")

    plot_comparison(args.var, args.dataperiod, args.mcperiod,
                    max_data_events=max_data_events,
                    max_mc_events=max_mc_events)

if __name__ == "__main__":
    main()
