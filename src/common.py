import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from rdkit.Chem import Mol
from skfp.bases import BaseFingerprintTransformer
from skfp.fingerprints import (
    ECFPFingerprint,
    RDKitFingerprint,
    TopologicalTorsionFingerprint,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline


def load_aminoseqs_from_file(filepath: str | Path) -> list[str]:
    aminosequences = []
    with open(filepath) as file:
        for line in file:
            if line.startswith(">") or not line.strip():
                continue
            aminosequences.append(line.strip())

    return aminosequences


def train_fp_model(
    fp_name: str,
    mols_train: list[Mol],
    y_train: np.ndarray,
    task: str,
    random_state: int = 0,
    count_fp: bool = True,
) -> Pipeline:
    # turn off unnecessary scikit-learn warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    fp_obj, _ = get_fp_by_name(fp_name, count=count_fp)
    clf = get_lightgbm(task, y_train, random_state)
    pipeline = Pipeline([("fp", fp_obj), ("clf", clf)])

    pipeline.fit(mols_train, y_train)
    return pipeline


def get_fp_by_name(
    fp_name: str, count: bool = True, sparse: bool = False
) -> tuple[BaseFingerprintTransformer, dict]:
    if fp_name == "ECFP":
        fp_cls = ECFPFingerprint(count=count, n_jobs=-1, sparse=sparse)
        params_grid = {"radius": [2, 3, 4]}
    elif fp_name == "TopologicalTorsion":
        fp_cls = TopologicalTorsionFingerprint(
            count=count, count_simulation=False, n_jobs=-1, sparse=sparse
        )
        params_grid = {"torsion_atom_count": [4, 6, 8]}
    elif fp_name == "RDKit":
        fp_cls = RDKitFingerprint(count=count, n_jobs=-1, sparse=sparse)
        params_grid = {"max_path": [7, 8, 9]}
    else:
        raise ValueError(f"fp_name {fp_name} not recognized")

    return fp_cls, params_grid


def get_lightgbm(
    task: str, y_train: np.ndarray, random_state: int
) -> LGBMClassifier | LGBMRegressor | MultiOutputClassifier | MultiOutputRegressor:
    num_tasks = 1 if y_train.ndim == 1 else y_train.shape[1]
    if task not in {"classification", "regression"}:
        raise ValueError(f"task {task} not recognized")

    if task == "classification":
        model = LGBMClassifier(
            n_estimators=500,
            is_unbalance=True,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )
        if num_tasks > 1:
            model = MultiOutputClassifier(model)
    elif task == "regression":
        model = LGBMRegressor(
            n_estimators=500,
            objective="mae",
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )
        if num_tasks > 1:
            model = MultiOutputRegressor(model)
    else:
        raise ValueError(f"task {task} not recognized")

    return model


def evaluate(model, mols_test, y_test) -> dict[str, float]:
    y_pred_proba = model.predict_proba(mols_test)[:, 1]
    y_pred = model.predict(mols_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "specificity": recall_score(y_test, y_pred, pos_label=0),
        "precision": precision_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "AUROC": roc_auc_score(y_test, y_pred_proba),
    }


def print_df_with_tabs(df: pd.DataFrame, line_beginning: str = "") -> None:
    print(line_beginning, "\t".join(df.columns), sep="")
    for idx, row in df.iterrows():
        row = [str(value) for value in row]
        print(line_beginning, "\t".join(row), sep="")
