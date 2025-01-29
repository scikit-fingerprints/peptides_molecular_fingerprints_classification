from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import Mol
from skfp.preprocessing import MolFromAminoseqTransformer

from src.common import (
    evaluate,
    load_aminoseqs_from_file,
    print_df_with_tabs,
    train_fp_model,
)


def load_dataset() -> tuple[list[Mol], list[Mol], np.ndarray, np.ndarray]:
    datasets_dir = Path("../data/Xu_AMP")

    mol_from_seq = MolFromAminoseqTransformer(n_jobs=-1)

    seqs_train_neg = load_aminoseqs_from_file(datasets_dir / "train_negative.fasta")
    seqs_train_pos = load_aminoseqs_from_file(datasets_dir / "train_positive.fasta")
    seqs_train = seqs_train_neg + seqs_train_pos

    seqs_test_neg = load_aminoseqs_from_file(datasets_dir / "test_negative.fasta")
    seqs_test_pos = load_aminoseqs_from_file(datasets_dir / "test_positive.fasta")
    seqs_test = seqs_test_neg + seqs_test_pos

    y_train = np.array([0] * len(seqs_train_neg) + [1] * len(seqs_train_pos))
    y_test = np.array([0] * len(seqs_test_neg) + [1] * len(seqs_test_pos))

    mols_train = mol_from_seq.transform(seqs_train)
    mols_test = mol_from_seq.transform(seqs_test)

    return mols_train, mols_test, y_train, y_test


if __name__ == "__main__":
    mols_train, mols_test, y_train, y_test = load_dataset()

    task = "classification"

    metrics_names = ["accuracy", "AUROC", "F1", "MCC", "recall", "specificity"]
    metrics_names_str = "\t".join(metrics_names)

    for fp_name in ["ECFP", "TopologicalTorsion", "RDKit"]:
        print(fp_name)
        model = train_fp_model(fp_name, mols_train, y_train, task)
        metrics = evaluate(model, mols_test, y_test)

        results = {name: metrics[name] for name in metrics_names}
        results = pd.DataFrame([results])
        results[metrics_names] = results[metrics_names].round(3)
        print_df_with_tabs(results)
        print("\n")
