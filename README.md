# Molecular Fingerprints Are Strong Models for Peptide Function Prediction

Code for paper "Molecular Fingerprints Are Strong Models for Peptide Function Prediction" J. Adamczyk, P. Ludynia, W. Czech,
from AGH ML & Chemoinformatics Group.

Preprint is available [on Arxiv](https://arxiv.org/abs/2501.17901).

### Setup

Dependencies are installed with Poetry. You can install exact versions used from `poetry.lock`
by running `poety sync` in fresh Python 3.11 virtual environment.

To ensure you have everything set up, you can also run `make setup`. All datasets used are
already included in the `data` directory.

### Main body experiments

To obtain results on benchmarks in the main paper body, run:
1. LRGB: `python src/main_lrgb.py`
2. LRGB binary fingerprints: `python src/main_lrgb_binary_fp.py_`
3. BERT AMPs benchmark of Gao et al.: `python src/main_bert_amp_benchmark.py`
4. AMPs benchmark of Xu et al.: `python src/main_xu_amp.py`
5. AMPs benchmark of Sidorczuk et al.:
   - run `python src/main_ampbenchmark.py`
   - result file will be created in `results` directory
   - upload it to [AMPBenchmark web server](https://biogenies.info/AMPBenchmark/)
   - download result from "Impact of the sampling method" tab
6. AutoPeptideML benchmark of Fernandez-Diaz et al.: `python src/main_autopeptideml.py`
7. PeptideReactor benchmark of Späanig et al.:
   - run `python src/main_peptidereactor.py`
   - run `python src/main_peptidereactor_fp_encoding.py`
   - result files will be created in `results` directory
   - parse the results by running `cd util_scripts && python parse_peptidereactor_results.py`

### Additional scripts

`util_scripts` directory contains a few utility scripts:
- `create_bert_benchmark_datasets.py` creates train-test dataset splits for BERT AMPs
  benchmark; requires CD-HIT to be installed
- `parse_autopeptideml_results.py` parses original results from AutoPeptideML, to create
  more readable CSV output
- `parse_peptidereactor_results.py` parses original results from PeptideReactor, which are
  a very complex JSON, and adds fingerprint results from training our models
