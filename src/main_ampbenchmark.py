import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import Mol
from skfp.bases import BaseFingerprintTransformer
from skfp.preprocessing import MolFromAminoseqTransformer
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

from src.common import get_fp_by_name, get_lightgbm

datasets_dir = Path("..", "data", "AMPBenchmark", "train_sequences")

submission_dir_path = Path("..", "results", "ampbenchmark", "submissions")

sampling_methods = [
    "Gabere&Noble",
    "dbAMP",
    "CS-AMPPred",
    "Wang-et-al",
    "Witten&Witten",
    "AMAP",
    "ampir-mature",
    "iAMP-2L",
    "AmpGram",
    "AMPScannerV2",
    "AMPlify",
]


def load_from_file(filepath: str | Path) -> [list[str], list[int]]:
    ids = []
    aminoseqs = []
    aminoseq_buffer = []
    labels = []
    with open(filepath, "r") as file:
        for line in file.readlines():
            if not line.startswith(">") and line.strip():
                aminoseq_buffer.append(line.strip())
                continue

            if aminoseq_buffer:
                aminoseqs.append("".join(aminoseq_buffer))
                aminoseq_buffer = []

            ids.append(line.strip()[1:])

            if "AMP=1" in line:
                labels.append(1)
            elif "AMP=0" in line:
                labels.append(0)
            else:
                raise ValueError("All id strings have to contain class information")

        if aminoseq_buffer:
            aminoseqs.append("".join(aminoseq_buffer))

    return np.array(ids), np.array(aminoseqs), np.array(labels)


batch_size = 1024


def load_test_data(fp_transformer: BaseFingerprintTransformer):
    indices, molecules, labels = load_from_file(
        Path("..", "data", "AMPBenchmark", "AMPBenchmark_public.fasta")
    )

    print("Converting test dataset to molecules")

    mol_from_seq = MolFromAminoseqTransformer(n_jobs=-1)
    pipeline = make_pipeline(mol_from_seq, fp_transformer)

    mols_transformed = []
    for pos in tqdm(range(0, len(molecules), batch_size)):
        mols_transformed.append(pipeline.transform(molecules[pos : pos + batch_size]))
    molecules = np.vstack(mols_transformed)
    del mols_transformed

    print("Spliting molecules into 5 parts (reps)")
    new_test_data, id_check = [], []
    for i in range(1, 6):
        idx = pd.Series(indices).str.contains(f"rep{i}").values
        id_check.append(indices[idx])
        new_test_data.append(molecules[idx])

    test_data = new_test_data

    if not np.all(indices == np.concatenate(id_check)):
        print("incorrect input label order in test file")
        exit()

    return test_data, labels, indices


def load_5_reps_dataset(
    method: str, fp_transformer: BaseFingerprintTransformer
) -> [str, tuple[list[Mol], np.ndarray]]:
    reps = []
    mol_from_seq = MolFromAminoseqTransformer(n_jobs=-1)
    print("    loading training data")
    for i in range(1, 6):
        file_name = f"Training_data_sampling_method={method}_rep{i}.fasta"
        _, aminoseq, classes = load_from_file(datasets_dir / file_name)
        aminoseq = mol_from_seq.transform(aminoseq)
        aminoseq = fp_transformer.transform(aminoseq)
        reps.append((aminoseq, classes))
    return reps


def get_prediction_probas(
    train_data: list[np.ndarray],
    test_data: [np.ndarray],
):
    probas, predictions = [], []
    for i, ((X_train, y_train), X_test) in enumerate(zip(train_data, test_data)):
        print(f"    rep{i + 1}")

        clf = get_lightgbm(task="classification", y_train=y_train)
        clf.fit(X_train, y_train)

        probas.append(clf.predict_proba(X_test)[:, 1])
        predictions.append(clf.predict(X_test))

    return np.concatenate(probas), np.concatenate(predictions)


if __name__ == "__main__":
    submission_dir_path.mkdir(exist_ok=True, parents=True)

    # turn off unnecessary scikit-learn warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Iterate over fingerprints first
    # Avoid multiple computations on test set
    for fp_name in ["ECFP", "TopologicalTorsion", "RDKit"]:
        print(f"Fingerprint: {fp_name}")

        fp_obj, _ = get_fp_by_name(fp_name)

        # Precompute test data fingerprints
        # Avoid storing molecule objects
        test_data_fp, all_labels, all_ids = load_test_data(fp_obj)

        df = None
        for method in sampling_methods:
            print(f"    {method}")

            # Load data
            train_data = load_5_reps_dataset(method, fp_obj)

            # Compute probas for storing in .csv submission
            # Predictions are only for printing scores
            probas, predictions = get_prediction_probas(train_data, test_data_fp)

            if predictions.shape != all_labels.shape:
                print(
                    f"invalid shapes: {predictions.shape} (should be {all_labels.shape})"
                )
                exit()

            # Print scores
            # those will not be saved in the csv files
            accuracy = accuracy_score(all_labels, predictions)
            auroc = roc_auc_score(all_labels, probas)
            mcc = matthews_corrcoef(all_labels, predictions)
            print(f"        accuracy: {accuracy:.2f}")
            print(f"        AUROC   : {auroc:.2f}")
            print(f"        MCC     : {mcc:.2f}")

            # Append new dataframe rows to the already existing ones
            new_rows = pd.DataFrame(
                {
                    "ID": all_ids,
                    "training_sampling": method,
                    "AMP_probability": probas,
                }
            )
            new_rows["AMP_probability"] = new_rows["AMP_probability"].round(4)

            if df is not None:
                df = pd.concat([df, new_rows], ignore_index=True)
            else:
                df = new_rows

        df.to_csv(
            submission_dir_path / f"{fp_name}.csv",
            index=False,
            quotechar='"',
            quoting=2,
        )
