#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pandas.errors import EmptyDataError

# Paths (match notebook)
ROOT = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(ROOT, "Dataset_ver2", "Data", "predataset")
mc_path = os.path.join(ROOT, "Dataset_ver3", "MC", "processed")

periods = ["A", "B", "C", "D", "E", "F", "G", "I", "K", "L"]
samples = ["Diboson", "Multijet", "Single_top", "ttbar", "Wjets", "Zjets"]

variables = {
    "pT_j1": (0, 2000, 50, r"$p_{T}^{j1}$ [GeV]", True),
    "pT_j2": (0, 1500, 50, r"$p_{T}^{j2}$ [GeV]", True),
    "eta_j1": (-3, 3, 50, r"$\\eta^{j1}$", False),
    "eta_j2": (-3, 3, 50, r"$\\eta^{j2}$", False),
    "m_jj": (0, 5000, 50, r"$m_{jj}$ [GeV]", True),
    "tau21_j1": (0, 1.5, 50, r"$\\tau_{21}^{j1}$", False),
    "tau21_j2": (0, 1.5, 50, r"$\\tau_{21}^{j2}$", False),
    "tau32_j1": (0, 1.5, 50, r"$\\tau_{32}^{j1}$", False),
    "tau32_j2": (0, 1.5, 50, r"$\\tau_{32}^{j2}$", False),
    "met": (0, 2000, 50, r"$E_{T}^{miss}$ [GeV]", True),
    "ht": (0, 5000, 50, r"$H_{T}$ [GeV]", True),
    "min_dPhi": (0, 3.2, 50, r"$\\min \\Delta \\phi(jet, MET)$", False),
}


def load_txt_safe(file_path, is_mc=False):
    if not os.path.isfile(file_path):
        print(f"Warning: file not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, delim_whitespace=True)
    except EmptyDataError:
        print(f"Warning: empty file: {file_path}")
        return None
    except Exception as e:
        print(f"Warning: failed to read {file_path}: {e}")
        return None

    if 'weight' not in df.columns:
        df['weight'] = 1.0
    return df


def collect_data_files():
    data_to_plot = []
    if not os.path.isdir(data_path):
        print(f"Warning: data path missing: {data_path}")
        return data_to_plot
    for period in periods:
        period_dir = os.path.join(data_path, period)
        if not os.path.isdir(period_dir):
            continue
        for run in os.listdir(period_dir):
            run_dir = os.path.join(period_dir, run)
            if not os.path.isdir(run_dir):
                continue
            for dataset in os.listdir(run_dir):
                data_to_plot.append(os.path.join(run_dir, dataset))
    return data_to_plot


def collect_mc_files():
    mc_to_plot = {s: [] for s in samples}
    if not os.path.isdir(mc_path):
        print(f"Warning: MC path missing: {mc_path}")
        return mc_to_plot
    for sample in samples:
        sample_dir = os.path.join(mc_path, sample)
        if not os.path.isdir(sample_dir):
            print(f"Warning: MC sample dir missing: {sample_dir}")
            continue
        for process in os.listdir(sample_dir):
            proc_dir = os.path.join(sample_dir, process)
            if not os.path.isdir(proc_dir):
                continue
            pattern = os.path.join(proc_dir, 'dataset_*.txt')
            found = glob.glob(pattern)
            if not found:
                # try any txt file
                found = glob.glob(os.path.join(proc_dir, '*.txt'))
            if found:
                mc_to_plot[sample].extend(found)
            else:
                print(f"Warning: no dataset files in {proc_dir}")
    return mc_to_plot


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    print("Starting kinematics plotting...")
    outdir = os.path.join(ROOT, 'plots', 'Data', 'kinematics')
    ensure_dir(outdir)

    # Collect and load data
    data_files = collect_data_files()
    data_list = [load_txt_safe(f, is_mc=False) for f in data_files]
    data_list = [d for d in data_list if d is not None]
    if data_list:
        data_df = pd.concat(data_list, ignore_index=True)
    else:
        cols = list(variables.keys()) + ['weight']
        data_df = pd.DataFrame(columns=cols)
        print("Warning: no data files loaded; continuing with empty DataFrame.")

    # Collect and load MC
    mc_files = collect_mc_files()
    mc_dfs = {}
    for mc_name, files in mc_files.items():
        dfs = [load_txt_safe(f, is_mc=True) for f in files]
        dfs = [d for d in dfs if d is not None]
        if dfs:
            mc_dfs[mc_name] = pd.concat(dfs, ignore_index=True)
        else:
            mc_dfs[mc_name] = pd.DataFrame(columns=list(variables.keys()) + ['weight'])

    # Plotting
    for var in variables.keys():
        vmin, vmax, nbins, xlabel, logy = variables[var]
        bins = np.linspace(vmin, vmax, nbins + 1)

        plt.figure(figsize=(8, 6))
        stacked_counts = np.zeros(nbins)
        colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(mc_dfs))))

        # Stack MC
        for i, (mc_name, df) in enumerate(mc_dfs.items()):
            if var not in df.columns or df.shape[0] == 0:
                # nothing to plot for this sample
                continue
            vals = pd.to_numeric(df[var], errors='coerce')
            mask = ~vals.isna()
            if mask.sum() == 0:
                continue
            vals = vals[mask].to_numpy()
            weights = df.loc[mask, 'weight'].to_numpy()
            counts, _ = np.histogram(vals, bins=bins, weights=weights)
            plt.bar(bins[:-1], counts, width=np.diff(bins), bottom=stacked_counts,
                    color=colors[i % len(colors)], alpha=0.7, edgecolor='black',
                    label=mc_name, align='edge')
            stacked_counts += counts

        # Data
        if var in data_df.columns and data_df.shape[0] > 0:
            data_vals = pd.to_numeric(data_df[var], errors='coerce')
            mask = ~data_vals.isna()
            data_vals = data_vals[mask].to_numpy()
            data_weights = data_df.loc[mask, 'weight'].to_numpy() if 'weight' in data_df.columns else np.ones_like(data_vals)
            data_counts, _ = np.histogram(data_vals, bins=bins, weights=data_weights)
            data_err, _ = np.histogram(data_vals, bins=bins, weights=data_weights**2)
            plt.errorbar((bins[:-1] + bins[1:]) / 2, data_counts, yerr=np.sqrt(data_err),
                         fmt='o', color='black', label='Data')
        else:
            print(f"Note: data missing or empty for variable {var}")

        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel('Events', fontsize=14)

        # Handle log-scale safely
        if logy:
            # small positive floor to avoid zeros when switching to log
            ymax = plt.ylim()[1]
            if ymax <= 0:
                plt.yscale('linear')
            else:
                plt.yscale('log')

        plt.legend(fontsize=10)
        plt.title(f'Distribution of {var}', fontsize=16)
        plt.tight_layout()

        outpath = os.path.join(outdir, f"{var}.png")
        plt.savefig(outpath)
        plt.close()
        print(f"Saved plot: {outpath}")


if __name__ == '__main__':
    main()
