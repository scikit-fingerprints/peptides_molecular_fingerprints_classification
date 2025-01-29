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

from src.common import train_fp_model


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


def evaluate(model, mols_test: list[Mol], y_test: np.ndarray, task: str) -> float:
    if task == "classification":
        y_pred_proba = model.predict_proba(mols_test)
        y_pred_proba = extract_pos_proba(y_pred_proba)
        return multioutput_auprc_score(y_test, y_pred_proba)
    else:
        y_pred = model.predict(mols_test)
        return multioutput_mean_absolute_error(y_test, y_pred)


if __name__ == "__main__":
    for fp_name in ["ECFP", "TopologicalTorsion", "RDKit"]:
        print(fp_name)

        mols_train, mols_test, y_train, y_test = get_train_test_splits("Peptides-func")
        task = "classification"
        model = train_fp_model(fp_name, mols_train, y_train, task)
        auprc = evaluate(model, mols_test, y_test, task)
        print(f"\tPeptides-func AUPRC {auprc:.4f}")

        mols_train, mols_test, y_train, y_test = get_train_test_splits(
            "Peptides-struct"
        )
        task = "regression"
        model = train_fp_model(fp_name, mols_train, y_train, task)
        mae = evaluate(model, mols_test, y_test, task)
        print(f"\tPeptides-struct MAE {mae:.4f}")
