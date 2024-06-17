Here are all the models' weights and results for models trained in low-data regimes.
* `extract.sh`
Call this script with list of ids to be ignored as the argument. This script was used to generate `MODEL_low_data.csv` summary files.

To reproduce paper results, call it as `extract.sh 4033 3392 939`.
* `analysis.ipynb` Notebook used to analyze the effects of sampling methods

Inside all the other folders, you will find the following strucutre:

_ mace_ffn \
__ version_0 \
__ version_1 \
__ version_2 \
__ version_3 \
__ version_4 \
__ version_5 \
__ version_6 \
__ mace_ffn_low_data.csv \
... \
_ extract.py \
Each main folder contains:
* `version_N` that contains one model's (trained on one sampling) checkpoints and config files and `test.csv` file that contains results on a test set.
* `MODEL_low_data.csv` summarized results for all low-data samplings for that model
`extract.py` the script used to generate `MODEL_low_data.csv`



