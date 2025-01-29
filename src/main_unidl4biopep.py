import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import Mol
from skfp.preprocessing import MolFromAminoseqTransformer
from sklearn.model_selection import train_test_split

from src.common import evaluate, print_df_with_tabs, train_fp_model


def load_datasets() -> list[
    tuple[str, tuple[list[str], list[str], np.ndarray, np.ndarray]]
]:
    datasets_dir = Path("../data/UniDL4BioPep")
    dataset_dir_names = sorted(os.listdir(datasets_dir))

    # read the 17 datasets with given train-test splits
    datasets = {}
    for dataset_dir_name in dataset_dir_names:
        # there is one unsplit dataset, we handle it manually below
        if "18. antioxidant" in dataset_dir_name:
            continue

        dataset_files = os.listdir(datasets_dir / dataset_dir_name)

        if "train" in dataset_files[0]:
            train_filename, test_filename = dataset_files
        else:
            test_filename, train_filename = dataset_files

        # there are peptides "NA", so we need to avoid reading them as np.nan
        df_train = pd.read_excel(
            datasets_dir / dataset_dir_name / train_filename,
            na_filter=False,
        )
        df_test = pd.read_excel(
            datasets_dir / dataset_dir_name / test_filename,
            na_filter=False,
        )

        seqs_train = df_train["sequence"]
        seqs_test = df_test["sequence"]

        y_train = df_train["label"].values
        y_test = df_test["label"].values

        # remove number, e.g. "3. Bitter" -> "Bitter"
        dataset_name = dataset_dir_name.split(". ")[1].strip()

        datasets[dataset_name] = (seqs_train, seqs_test, y_train, y_test)

    # we need to split the antioxidant dataset ourselves
    filepath = datasets_dir / "18. antioxidant_FRS" / "antioxidant_dataset.xlsx"
    df = pd.read_excel(filepath)

    # in UniDL4BioPep code, test_size=0.2 and random_state=123 are used everywhere
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=123)

    seqs_train = df_train["sequence"]
    seqs_test = df_test["sequence"]

    y_train = df_train["label"]
    y_test = df_test["label"]

    datasets["antioxidant_FRS"] = seqs_train, seqs_test, y_train, y_test

    # reorder datasets, to keep the same order as the paper
    dataset_names = [
        "ACE inhibitory activity",
        "DPPIV inhibitory activity",
        "Bitter",
        "Umami",
        "Antimicrobial activity",
        "Antimalarial activity-main",
        "Antimalarial activity-alternative",
        "Quorum sensing activity",
        "ACP Anticancer activity-main",
        "ACP Anticancer activity-alternative",
        "Anti-MRSA strains activity",
        "TTCA",
        "BBP Blood-Brain Barrier Peptides",
        "APP  Anti-parasitic",
        "NeuroPred",
        "antibacterial AB",
        "Antifungal AF",
        "AV Antiviral",
        "Toxicity 2021 Dataset",
        "antioxidant_FRS",
    ]
    datasets = [
        (dataset_name, datasets[dataset_name]) for dataset_name in dataset_names
    ]

    return datasets


def seqs_to_mols(seqs: list[str], y: np.ndarray) -> tuple[list[Mol], np.ndarray]:
    mol_from_seq = MolFromAminoseqTransformer(n_jobs=-1)
    mols = mol_from_seq.transform(seqs)

    # filter out molecules that errored out, there is one or two in the entire benchmark
    valid_mols_mask = [mol is not None for mol in mols]
    mols = np.array(mols)[valid_mols_mask].tolist()
    y = y[valid_mols_mask]

    return mols, y


if __name__ == "__main__":
    datasets = load_datasets()

    metrics_names = [
        "accuracy",
        "balanced_accuracy",
        "recall",
        "specificity",
        "MCC",
        "AUROC",
    ]
    metrics_names_str = "\t".join(metrics_names)

    for fp_name in ["ECFP", "TopologicalTorsion", "RDKit"]:
        print(fp_name)
        results = []
        for dataset_name, (seqs_train, seqs_test, y_train, y_test) in datasets:
            print(f"{dataset_name}")
            mols_train, y_train = seqs_to_mols(seqs_train, y_train)
            mols_test, y_test = seqs_to_mols(seqs_test, y_test)

            model = train_fp_model(fp_name, mols_train, y_train, task="classification")
            metrics = evaluate(model, mols_test, y_test)

            result = {name: metrics[name] for name in metrics_names}
            result["dataset_name"] = dataset_name
            results.append(result)

        results = pd.DataFrame(results)
        results[metrics_names] = results[metrics_names].round(3)
        results = results[["dataset_name"] + metrics_names]
        print_df_with_tabs(results)
        print("\n")
