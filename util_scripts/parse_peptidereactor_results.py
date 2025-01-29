import json

import pandas as pd

"""
PeptideReactor results are originally a huge JSON. Here, we transform it into
a tabular form, and then compare results per dataset.
"""

if __name__ == "__main__":
    with open("data/peptidereactor_results.json") as file:
        data = json.load(file)

    df_paper = pd.DataFrame.from_records(data)
    df_paper = df_paper[["Dataset", "Encoding", "F1"]]
    df_paper = df_paper[(df_paper["F1"] != "separator") & (~df_paper["F1"].isna())]

    # add fingerprint results
    df_ours = pd.read_csv("../results/peptidereactor.csv")
    df = pd.concat([df_paper, df_ours])

    dfs_with_ranks = []
    for dataset in df["Dataset"].unique():
        df_dataset = df[df["Dataset"] == dataset].copy()
        df_dataset["rank"] = df_dataset["F1"].rank(ascending=False)
        dfs_with_ranks.append(df_dataset)

    df_ranks = pd.concat(dfs_with_ranks)
    df_ranks = df_ranks[["Encoding", "rank"]]

    fps = [
        "ECFP",
        "TT",
        "RDKit",
        "ECFP tuned",
        "TT tuned",
        "RDKit tuned",
        "FP encoding",
    ]
    df_fp = df_ranks[df_ranks["Encoding"].isin(fps)]
    num_sota_datasets = (df_fp["rank"] == 1).sum()
    print(f"Fingerprints are SOTA on {num_sota_datasets} datasets")

    for fp in fps:
        mean = df_ours.loc[df_ours["Encoding"] == fp, "F1"].mean()
        std = df_ours.loc[df_ours["Encoding"] == fp, "F1"].std()
        print(f"{fp} mean F1: {mean:.3f} +- {std:.3f}")

    df_avg_ranks = df_ranks.groupby("Encoding").mean()
    df_avg_ranks = df_avg_ranks.sort_values(by="rank").reset_index()

    print("\n")
    print("Encoding\tAvg rank")
    for encoding, rank in zip(df_avg_ranks["Encoding"], df_avg_ranks["rank"]):
        print(encoding, "\t", rank)

    df = df[["Encoding", "F1"]]
    df["F1"] = df["F1"].astype(float)

    df_avg_f1 = df.groupby("Encoding").mean()
    df_std_f1 = df.groupby("Encoding").std()
    df_f1 = pd.merge(df_avg_f1, df_std_f1, on="Encoding", suffixes=("_mean", "_std"))
    df_f1 = df_f1.sort_values(by="F1_mean", ascending=False).round(3).reset_index()

    print("\n")
    print("Encoding\tAvg F1")
    for idx, row in df_f1.iterrows():
        encoding = row["Encoding"]
        f1_mean = row["F1_mean"]
        f1_std = row["F1_std"]
        print(encoding, "\t", f1_mean, "+-", f1_std)
