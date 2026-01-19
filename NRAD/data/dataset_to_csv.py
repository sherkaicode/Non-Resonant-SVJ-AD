#!/usr/bin/env python3
"""
Read processed MC `dataset_*.txt` files and write per-process CSVs and a combined Full_MC_Data.csv

Usage:
  python3 dataset_to_csv.py

This script expects the processed datasets under:
  Dataset_ver3/MC/processed/<process>/**/dataset_*.txt

And writes CSVs to:
  NRAD/data/MC_<process>.csv and NRAD/data/Full_MC_Data.csv
"""
import os
import glob
import pandas as pd
import argparse


def process_to_csv(process_dir, out_dir):
    process = os.path.basename(process_dir.rstrip('/'))
    print(f"Processing process: {process}")
    pattern = os.path.join(process_dir, '**', 'dataset_*.txt')
    files = glob.glob(pattern, recursive=True)
    print(f"  Found {len(files)} dataset files")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, delim_whitespace=True)
        except Exception as e:
            print(f"  Warning: failed to read {f}: {e}")
            continue
        # add metadata columns
        df['sample'] = process
        df['source'] = os.path.basename(os.path.dirname(f))
        dfs.append(df)

    if not dfs:
        print(f"  No readable datasets for {process}, skipping output.")
        return None

    out_df = pd.concat(dfs, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f"MC_{process}.csv")
    out_df.to_csv(outfile, index=False)
    print(f"  Wrote {outfile} ({len(out_df):,} rows)")
    return outfile


def main():
    parser = argparse.ArgumentParser(description='Convert processed dataset_*.txt to CSVs')
    parser.add_argument('--mc-root', default=os.path.join('..', '..', 'Dataset_ver3', 'MC', 'processed'),
                        help='Root directory containing processed MC (default: Dataset_ver3/MC/processed)')
    parser.add_argument('--out-dir', default='.', help='Output directory for CSVs (default: NRAD/data)')
    parser.add_argument('--full', action='store_true', help='Also write Full_MC_Data.csv concatenating all processes')
    args = parser.parse_args()

    mc_root = os.path.abspath(os.path.join(os.path.dirname(__file__), args.mc_root))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.out_dir))

    # Ensure output dir exists
    os.makedirs(out_dir, exist_ok=True)

    # Find processes
    if not os.path.isdir(mc_root):
        print(f"MC root not found: {mc_root}")
        return

    processes = sorted([p for p in glob.glob(os.path.join(mc_root, '*')) if os.path.isdir(p)])
    written_files = []
    for p in processes:
        out = process_to_csv(p, out_dir)
        if out:
            written_files.append(out)

    if args.full and written_files:
        print("Building Full_MC_Data.csv by concatenating per-process CSVs...")
        dfs = []
        for fn in written_files:
            try:
                df = pd.read_csv(fn)
                dfs.append(df)
            except Exception as e:
                print(f"  Warning: failed to read {fn}: {e}")
        if dfs:
            full = pd.concat(dfs, ignore_index=True)
            full_out = os.path.join(out_dir, 'Full_MC_Data.csv')
            full.to_csv(full_out, index=False)
            print(f"Wrote {full_out} ({len(full):,} rows)")


if __name__ == '__main__':
    main()
