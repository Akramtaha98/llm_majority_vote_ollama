from __future__ import annotations
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.preds)
    acc = df["correct"].mean() * 100.0
    plt.figure(figsize=(5,3.2))
    plt.bar(["Accuracy"], [acc])
    plt.ylabel("Accuracy (%)")
    plt.title("Majority Voting Performance")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
