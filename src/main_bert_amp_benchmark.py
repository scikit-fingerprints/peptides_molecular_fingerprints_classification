from pathlib import Path

import numpy as np
import pandas as pd
from skfp.preprocessing import MolFromAminoseqTransformer

from src.common import evaluate, print_df_with_tabs, train_fp_model

BERT_AMP_BENCHMARK_DATASET_NAMES = [
    "ADAPTABLE",
    "APD",
    "CAMP",
    "dbAMP",
    "DRAMP",
    "YADAMP",
]


def load_datasets() -> dict[str, tuple[list[str], list[str], np.ndarray, np.ndarray]]:
    datasets_dir = Path("../data/BERT_AMP_benchmark")

    datasets = {}
    for name in BERT_AMP_BENCHMARK_DATASET_NAMES:
        train_filepath = datasets_dir / "preprocessed_datasets" / f"{name}_train.fasta"
        test_filepath = datasets_dir / "preprocessed_datasets" / f"{name}_test.fasta"

        seqs_train, y_train = load_aminoseqs_from_file(train_filepath)
        seqs_test, y_test = load_aminoseqs_from_file(test_filepath)

        datasets[name] = (seqs_train, seqs_test, y_train, y_test)

    return datasets


def load_aminoseqs_from_file(filepath: str | Path) -> tuple[list[str], np.ndarray]:
    seqs_pos = []
    seqs_neg = []
    with open(filepath) as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith(">") and "for AMP" in line:
                seqs_pos.append(lines[i + 1].strip())
            elif line.startswith(">") and "for NAMP" in line:
                seqs_neg.append(lines[i + 1].strip())
            i += 2

    y_pos = np.ones(len(seqs_pos))
    y_neg = np.zeros(len(seqs_neg))

    seqs = seqs_pos + seqs_neg
    y = np.concatenate((y_pos, y_neg))

    return seqs, y


if __name__ == "__main__":
    datasets = load_datasets()

    mol_from_seq = MolFromAminoseqTransformer(n_jobs=-1)

    metrics_names = ["recall", "precision", "AUROC", "F1"]
    metrics_names_str = "\t".join(metrics_names)

    for fp_name in ["ECFP", "TopologicalTorsion", "RDKit"]:
        print(f"Processing {fp_name} fingerprint")
        results = []

        for current_dataset_name in BERT_AMP_BENCHMARK_DATASET_NAMES:
            print(f"{current_dataset_name}")

            seqs_train, seqs_test, y_train, y_test = datasets[current_dataset_name]

            mols_train = mol_from_seq.transform(seqs_train)
            mols_test = mol_from_seq.transform(seqs_test)

            model = train_fp_model(fp_name, mols_train, y_train, task="classification")
            metrics = evaluate(model, mols_test, y_test)

            result = {name: metrics[name] for name in metrics_names}
            result["dataset_name"] = current_dataset_name
            results.append(result)

        results = pd.DataFrame(results)
        results[metrics_names] = results[metrics_names].round(3)
        results = results[["dataset_name"] + metrics_names]
        print_df_with_tabs(results)
        print("\n")
