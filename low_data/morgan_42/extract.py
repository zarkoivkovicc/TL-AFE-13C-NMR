from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import os
import sys

models = os.listdir(".")
for f in ["extract.py", "analysis.ipynb"]:
    models.remove(f)
NUM_TRAIN_EXAMPLES = [100, 250, 500, 1000, 2500, 5000, 10000]
id_erroneous = [int(i) for i in sys.argv[1:]]
for model in models:
    maes = []
    rmses = []
    for i in range(7):
        df = pd.read_csv(f"{model}/version_{i}/test.csv")
        df = df[~df["mol_idx"].isin(id_erroneous)]
        MAE = mean_absolute_error(df["true_shift"], df["predicted_shift"])
        RMSE = np.sqrt(mean_squared_error(df["true_shift"], df["predicted_shift"]))
        maes.append(MAE)
        rmses.append(RMSE)
    results = pd.DataFrame(
        {"num_train_examples": NUM_TRAIN_EXAMPLES, "MAE": maes, "RMSE": rmses}
    )
    print(model)
    print(results)
    results.to_csv(f"{model}/{model}_low_data.csv")
