import os.path
import shutil
import subprocess

"""
Based on the original datasets from BERT benchmark, create a fair "pretraining" datasets
for fingerprints: https://github.com/AhauBioinformatics/AMPpred-BERT-assessment.

Uses leave-one-out-dataset strategy, where:
- one dataset is the test set
- all others are merged into training set
- CD-HIT with threshold 40% is used to remove homology bias by filtering the training set
"""


def load_dataset_fasta(dataset_name: str) -> str:
    with open(
        f"../data/BERT_AMP_benchmark/original_datasets/{dataset_name}/{dataset_name.lower()}_amps.fa"
    ) as file:
        fasta_pos = [line.strip() for line in file.readlines() if line.strip()]
        fasta_pos = "\n".join(fasta_pos)

    with open(
        f"../data/BERT_AMP_benchmark/original_datasets/{dataset_name}/{dataset_name.lower()}_nonamps.fa"
    ) as file:
        fasta_neg = [line.strip() for line in file.readlines() if line.strip()]
        fasta_neg = "\n".join(fasta_neg)

    fasta = fasta_pos + "\n" + fasta_neg
    return fasta


def run_cdhit_train_filtering(train_fasta: str, test_fasta: str) -> str:
    tmp_dir_name = "tmp"
    if os.path.exists(tmp_dir_name):
        shutil.rmtree(tmp_dir_name)
    os.mkdir(tmp_dir_name)

    with open(os.path.join(tmp_dir_name, "train.fa"), "w") as file:
        file.write(train_fasta)

    with open(os.path.join(tmp_dir_name, "test.fa"), "w") as file:
        file.write(test_fasta)

    # fmt: off
    subprocess.run([
        "cd-hit-2d",
        "-i", f"{tmp_dir_name}/test.fa",
        "-i2", f"{tmp_dir_name}/train.fa",
        "-o", f"{tmp_dir_name}/train_filtered.fa",
        "-c", "0.4",
        "-n", "2",
        "-T", "0",
    ])
    # fmt: on

    with open(f"{tmp_dir_name}/train_filtered.fa") as file:
        train_fasta = file.read()

    return train_fasta


if __name__ == "__main__":
    dataset_names = [
        "ADAPTABLE",
        "APD",
        "CAMP",
        "dbAMP",
        "DRAMP",
        "YADAMP",
    ]

    datasets = {
        dataset_name: load_dataset_fasta(dataset_name) for dataset_name in dataset_names
    }
    datasets_filtered = {}

    output_dir_name = "bert_benchmark_datasets_outputs"
    if os.path.exists(output_dir_name):
        shutil.rmtree(output_dir_name)
    os.mkdir(output_dir_name)

    for curr_dataset_name, test_fasta in datasets.items():
        train_fasta_list = [
            fasta
            for dataset_name, fasta in datasets.items()
            if dataset_name != curr_dataset_name
        ]
        train_fasta = "\n".join(train_fasta_list)
        train_fasta = run_cdhit_train_filtering(train_fasta, test_fasta)

        with open(f"{output_dir_name}/{curr_dataset_name}_train.fasta", "w") as file:
            file.write(train_fasta)

        with open(f"{output_dir_name}/{curr_dataset_name}_test.fasta", "w") as file:
            file.write(test_fasta)
