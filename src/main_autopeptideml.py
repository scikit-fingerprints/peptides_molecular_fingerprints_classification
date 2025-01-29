import os
from pathlib import Path

import numpy as np
import pandas as pd
from skfp.preprocessing import MolFromAminoseqTransformer

from src.common import evaluate, train_fp_model


def load_datasets() -> dict[str, tuple[list[str], list[str], np.ndarray, np.ndarray]]:
    datasets_dir = Path("..", "data", "AutoPeptideML")
    dataset_names = sorted(os.listdir(datasets_dir))

    datasets = {}
    for name in dataset_names:
        train_filepath = datasets_dir / name / "splits" / "train.csv"
        test_filepath = datasets_dir / name / "splits" / "test.csv"

        df_train = pd.read_csv(train_filepath)
        df_test = pd.read_csv(test_filepath)

        seqs_train = df_train["sequence"]
        seqs_test = df_test["sequence"]

        y_train = df_train["Y"]
        y_test = df_test["Y"]

        datasets[name] = (seqs_train, seqs_test, y_train, y_test)

    return datasets


if __name__ == "__main__":
    datasets = load_datasets()

    mol_from_seq = MolFromAminoseqTransformer(n_jobs=-1)

    for fp_name in ["ECFP", "TopologicalTorsion", "RDKit"]:
        all_metrics = []
        for dataset_name, (seqs_train, seqs_test, y_train, y_test) in datasets.items():
            print(dataset_name)

            mols_train = mol_from_seq.transform(seqs_train)
            mols_test = mol_from_seq.transform(seqs_test)

            model = train_fp_model(
                fp_name,
                mols_train,
                y_train,
                task="classification",
            )
            metrics = evaluate(model, mols_test, y_test)

            for name, value in metrics.items():
                print(f"\t{name}: {value:.3f}")
            print()

            all_metrics.append(metrics)

        print("Overall metrics:")
        all_metrics = pd.DataFrame(all_metrics)
        for name in all_metrics:
            mean = all_metrics[name].mean()
            std = all_metrics[name].std()
            print(f"{name}: {mean:.3f} +- {std:.3f}")
