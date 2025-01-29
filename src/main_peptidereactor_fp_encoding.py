import os
from pathlib import Path

import numpy as np
import pandas as pd
from skfp.preprocessing import MolFromAminoseqTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from src.common import get_fp_by_name, load_aminoseqs_from_file


def load_datasets() -> dict[str, tuple[list[str], np.ndarray]]:
    datasets_dir = Path("../data/PeptideReactor")
    dataset_names = sorted(os.listdir(datasets_dir))

    datasets = {}
    for name in dataset_names:
        seqs_filepath = datasets_dir / name / "seqs.fasta"
        seqs = load_aminoseqs_from_file(seqs_filepath)

        labels_filepath = datasets_dir / name / "classes.txt"
        labels = pd.read_csv(labels_filepath, index_col=False, header=None)
        labels = labels.values.ravel()

        datasets[name] = (seqs, labels)

    return datasets


def get_repeated_kfold_score(mols: np.array, y: np.array) -> float:
    f1_values = []

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    for i, (train_index, test_index) in enumerate(cv.split(mols, y)):
        mols_train = mols[train_index]
        mols_test = mols[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # in general fingerprint encoding, we check all fingerprints and select
        # the one with the highest score
        best_pipeline = None
        best_score = -1
        for fp_name in ["ECFP", "TopologicalTorsion", "RDKit"]:
            fp_pipeline, fp_score = get_fp_pipeline(mols_train, y_train, fp_name)
            if fp_score > best_score:
                best_score = fp_score
                best_pipeline = fp_pipeline

        y_pred = best_pipeline.predict(mols_test)

        f1 = f1_score(y_test, y_pred)
        f1_values.append(f1)

    # PeptideReactor reports median metric over 50 measured test values
    median_f1 = np.median(f1_values)
    return median_f1


def get_fp_pipeline(
    mols_train: np.array, y_train: np.array, fp_name: str
) -> tuple[Pipeline, float]:
    fp_cls, fp_params_grid = get_fp_by_name(fp_name)

    # this paper uses RF with 100 trees for all encodings
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

    pipeline = Pipeline([("fp", fp_cls), ("clf", clf)])
    params_grid = {f"fp__{k}": v for k, v in fp_params_grid.items()}
    cv = GridSearchCV(
        estimator=pipeline,
        param_grid=params_grid,
        scoring=make_scorer(matthews_corrcoef),
        cv=5,
    )
    cv.fit(mols_train, y_train)

    return cv.best_estimator_, cv.best_score_


if __name__ == "__main__":
    if not os.path.exists("../results"):
        os.mkdir("../results")

    datasets = load_datasets()

    mol_from_seq = MolFromAminoseqTransformer(n_jobs=-1)

    # results in .csv format can be compared to original paper with script in:
    # util_scripts/parse_peptidereactor_results.py
    results = []

    f1_values = []
    for dataset_name, (seqs, labels) in datasets.items():
        mols = mol_from_seq.transform(seqs)
        mols = np.array(mols)

        f1 = get_repeated_kfold_score(mols, labels)
        f1_values.append(f1)

        results.append({"Dataset": dataset_name, "Encoding": "FP encoding", "F1": f1})
        print(f"{dataset_name}\t{f1}")

    f1 = np.mean(f1_values)
    print(f"FP encoding average F1: {f1}")
    print()

    df = pd.DataFrame.from_records(results)
    filepath = "../results/peptidereactor_fp_encoding.csv"
    with open(filepath, "a+") as file:
        header = not os.path.exists(filepath)
        df.to_csv(file, index=False, header=header)
