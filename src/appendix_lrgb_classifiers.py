import warnings

import numpy as np
from rdkit.Chem import Mol
from skfp.datasets.lrgb import (
    load_lrgb_mol_splits,
    load_peptides_func,
    load_peptides_struct,
)
from skfp.metrics import (
    extract_pos_proba,
    multioutput_auprc_score,
    multioutput_mean_absolute_error,
)
from skfp.preprocessing import MolFromSmilesTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from tqdm import tqdm

from src.common import get_fp_by_name


def load_datasets() -> dict[str, tuple[list[Mol], list[Mol], np.ndarray, np.ndarray]]:
    datasets = {
        "Peptides-func": get_train_test_splits("Peptides-func"),
        "Peptides-struct": get_train_test_splits("Peptides-struct"),
    }
    return datasets


def get_train_test_splits(
    dataset_name: str,
) -> tuple[list[Mol], list[Mol], np.ndarray, np.ndarray]:
    if dataset_name == "Peptides-func":
        loader_func = load_peptides_func
    else:
        loader_func = load_peptides_struct

    smiles, y = loader_func()
    train_idx, valid_idx, test_idx = load_lrgb_mol_splits(dataset_name)

    mol_from_smiles = MolFromSmilesTransformer(n_jobs=-1)
    mols = mol_from_smiles.transform(smiles)
    mols = np.array(mols)

    mols_train = list(mols[train_idx + valid_idx])
    mols_test = list(mols[test_idx])

    y_train = y[train_idx + valid_idx]
    y_test = y[test_idx]

    return mols_train, mols_test, y_train, y_test


def train_model(
    fp_name: str,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
    random_state: int = 0,
):
    # turn off unnecessary scikit-learn warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    fp_obj, _ = get_fp_by_name(fp_name)

    if model_name == "RandomForest":
        if task == "classification":
            model = RandomForestClassifier(
                n_estimators=500,
                criterion="entropy",
                n_jobs=-1,
                random_state=random_state,
                class_weight="balanced",
            )
        else:
            model = RandomForestRegressor(
                n_estimators=500,
                criterion="absolute_error",
                n_jobs=-1,
                random_state=random_state,
            )
    elif model_name == "ExtraTrees":
        if task == "classification":
            model = ExtraTreesClassifier(
                n_estimators=500,
                criterion="entropy",
                n_jobs=-1,
                random_state=random_state,
                class_weight="balanced",
            )
        else:
            model = ExtraTreesRegressor(
                n_estimators=500,
                criterion="absolute_error",
                n_jobs=-1,
                random_state=random_state,
            )

    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, task: str) -> float:
    if task == "classification":
        y_pred_proba = model.predict_proba(X_test)
        y_pred_proba = extract_pos_proba(y_pred_proba)
        return multioutput_auprc_score(y_test, y_pred_proba)
    else:
        y_pred = model.predict(X_test)
        return multioutput_mean_absolute_error(y_test, y_pred)


if __name__ == "__main__":
    for model_name in ["RandomForest", "ExtraTrees"]:
        print(model_name)
        for fp_name in ["ECFP", "TopologicalTorsion", "RDKit"]:
            print(fp_name)

            mols_train, mols_test, y_train, y_test = get_train_test_splits(
                "Peptides-func"
            )
            fp_obj, _ = get_fp_by_name(fp_name)
            X_train = fp_obj.transform(mols_train)
            X_test = fp_obj.transform(mols_test)

            auprc_values = []
            for random_state in tqdm(range(10), total=10):
                task = "classification"
                model = train_model(
                    fp_name, model_name, X_train, y_train, task, random_state
                )
                auprc = evaluate(model, X_test, y_test, task)
                auprc_values.append(auprc)

            auprc_mean = np.mean(auprc_values)
            auprc_std = np.std(auprc_values)
            print(f"\tPeptides-func AUPRC {auprc_mean:.4f} +- {auprc_std:.4f}")

            mols_train, mols_test, y_train, y_test = get_train_test_splits(
                "Peptides-struct"
            )
            fp_obj, _ = get_fp_by_name(fp_name)
            X_train = fp_obj.transform(mols_train)
            X_test = fp_obj.transform(mols_test)

            mae_values = []
            for random_state in tqdm(range(10), total=10):
                task = "regression"
                model = train_model(
                    fp_name, model_name, X_train, y_train, task, random_state
                )
                mae = evaluate(model, X_test, y_test, task)
                mae_values.append(mae)

            mae_mean = np.mean(mae_values)
            mae_std = np.std(mae_values)
            print(f"\tPeptides-struct MAE {mae_mean:.4f} +- {mae_std:.4f}")
