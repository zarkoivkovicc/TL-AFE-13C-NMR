# Script to sample low-data regimes
# Warning: We didn't optimize any step here
# Sampling based on Hausdorff distance takes 24h !!!
from preprocessing.sampling import ActiveSampler
import numpy as np

BAD_EXAMPLES = [21501, 16095]
train_dataset_sizes = np.array([100, 250, 500, 1000, 2500, 5000, 10000])
total_dataset_sizes = [int(i) for i in train_dataset_sizes * 6 / 5]

train_ids_mace = {}
train_ids_unimol = {}

al = ActiveSampler("NMR_FF_train_unimol_C", sdf="NMR_FF_train")
for i in total_dataset_sizes:
    al.sample_dataset(based_on="random", n=i, seed=42, suffix="unimol_C")
    train_ids_unimol[i] = al.train_ids.copy()
al = ActiveSampler("NMR_FF_train_unimol", sdf="NMR_FF_train")
for i in total_dataset_sizes:
    al.train_ids = train_ids_unimol[i]
    al.save_dataset(f"random_42/{i}_unimol")

al = ActiveSampler("NMR_FF_train_mace_l_C", sdf="NMR_FF_train")
for i in total_dataset_sizes:
    al.sample_dataset(based_on="random", n=i, seed=42, suffix="mace_l_C")
    train_ids_mace[i] = al.train_ids.copy()
al = ActiveSampler("NMR_FF_train_mace_l", sdf="NMR_FF_train")
for i in total_dataset_sizes:
    al.train_ids = train_ids_mace[i]
    al.save_dataset(f"random_42/{i}_mace_l")

train_ids_mace = {}
train_ids_unimol = {}

al = ActiveSampler("NMR_FF_train_unimol_C", sdf="NMR_FF_train", ignore=BAD_EXAMPLES)
for i in total_dataset_sizes:
    al.sample_dataset(based_on="morgan", n=i, seed=42, suffix="unimol_C")
    train_ids_unimol[i] = al.train_ids.copy()
al = ActiveSampler("NMR_FF_train_unimol", sdf="NMR_FF_train", ignore=BAD_EXAMPLES)
for i in total_dataset_sizes:
    al.train_ids = train_ids_unimol[i]
    al.save_dataset(f"morgan_42/{i}_unimol")

al = ActiveSampler("NMR_FF_train_mace_l_C", sdf="NMR_FF_train", ignore=BAD_EXAMPLES)
for i in total_dataset_sizes:
    al.sample_dataset(based_on="morgan", n=i, seed=42, suffix="mace_l_C")
    train_ids_mace[i] = al.train_ids.copy()
al = ActiveSampler("NMR_FF_train_mace_l", sdf="NMR_FF_train", ignore=BAD_EXAMPLES)
for i in total_dataset_sizes:
    al.train_ids = train_ids_mace[i]
    al.save_dataset(f"morgan_42/{i}_mace_l")

train_ids_mace = {}
train_ids_unimol = {}

al = ActiveSampler("NMR_FF_train_unimol_C", sdf="NMR_FF_train", ignore=BAD_EXAMPLES)
for i in total_dataset_sizes:
    al.sample_dataset(based_on="euclidian_mean_encodings", n=i, seed=42, suffix="unimol_C")
    train_ids_unimol[i] = al.train_ids.copy()
al = ActiveSampler("NMR_FF_train_unimol", sdf="NMR_FF_train", ignore=BAD_EXAMPLES)
for i in total_dataset_sizes:
    al.train_ids = train_ids_unimol[i]
    al.save_dataset(f"euclidian_mean_encodings_42/{i}_unimol")

al = ActiveSampler("NMR_FF_train_mace_l_C", sdf="NMR_FF_train", ignore=BAD_EXAMPLES)
for i in total_dataset_sizes:
    al.sample_dataset(based_on="euclidian_mean_encodings", n=i, seed=42, suffix="mace_l_C")
    train_ids_mace[i] = al.train_ids.copy()
al = ActiveSampler("NMR_FF_train_mace_l", sdf="NMR_FF_train", ignore=BAD_EXAMPLES)
for i in total_dataset_sizes:
    al.train_ids = train_ids_mace[i]
    al.save_dataset(f"euclidian_mean_encodings_42/{i}_mace_l")

train_ids_mace = {}
train_ids_unimol = {}

al = ActiveSampler("NMR_FF_train_unimol_C", sdf="NMR_FF_train", ignore=BAD_EXAMPLES)
for i in total_dataset_sizes:
    al.sample_dataset(based_on="hausdorff_encodings", n=i, seed=42, suffix="unimol_C")
    train_ids_unimol[i] = al.train_ids.copy()
al = ActiveSampler("NMR_FF_train_unimol", sdf="NMR_FF_train", ignore=BAD_EXAMPLES)
for i in total_dataset_sizes:
    al.train_ids = train_ids_unimol[i]
    al.save_dataset(f"hausdorff_encodings_42/{i}_unimol")

al = ActiveSampler("NMR_FF_train_mace_l_C", sdf="NMR_FF_train", ignore=BAD_EXAMPLES)
for i in total_dataset_sizes:
    al.sample_dataset(based_on="hausdorff_encodings", n=i, seed=42, suffix="mace_l_C")
    train_ids_mace[i] = al.train_ids.copy()
al = ActiveSampler("NMR_FF_train_mace_l", sdf="NMR_FF_train", ignore=BAD_EXAMPLES)
for i in total_dataset_sizes:
    al.train_ids = train_ids_mace[i]
    al.save_dataset(f"hausdorff_encodings_42/{i}_mace_l")
