import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import directed_hausdorff, euclidean
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from rdkit.Chem import rdFingerprintGenerator, SDMolSupplier
from typing import Callable
from pathlib import Path


class ActiveSampler(object):
    def __init__(
        self, dataset: str, sdf: str | None = None, ignore: list[int] | None = None
    ):
        self.dataset = pd.read_parquet(f"data/atomic_datasets/{dataset}.parquet")
        if sdf:
            self.sdf = list(SDMolSupplier(f"raw_data/{sdf}.sdf", True, False, True))
        self.train_ids = []
        if ignore:
            self.dataset = self.dataset[~self.dataset["mol_idx"].isin(ignore)]
            self.sdf[:] = [x for x in self.sdf if x not in ignore]

    def save_dataset(self, filename: str):
        dir = Path("data/AL_sampled/") / Path(filename).parent
        dir.mkdir(parents=True, exist_ok=True)
        self.dataset[self.dataset["mol_idx"].isin(self.train_ids)].to_parquet(
            f"data/AL_sampled/{filename}.parquet"
        )

    def distance_hausdorff_encodings(self, i: int, j: int):
        encodings_i = np.stack(
            self.dataset[self.dataset["mol_idx"] == self.id_mapping[i]][
                "encoding"
            ].values,
            axis=1,
        ).T
        encodings_j = np.stack(
            self.dataset[self.dataset["mol_idx"] == self.id_mapping[j]][
                "encoding"
            ].values,
            axis=1,
        ).T
        return np.max(
            [
                directed_hausdorff(encodings_i, encodings_j),
                directed_hausdorff(encodings_j, encodings_i),
            ]
        )

    def distance_euclidian_mean_encodings(self, i: int, j: int):
        encodings_i = np.stack(
            self.dataset[self.dataset["mol_idx"] == self.id_mapping[i]][
                "encoding"
            ].values,
            axis=1,
        ).T
        encodings_j = np.stack(
            self.dataset[self.dataset["mol_idx"] == self.id_mapping[j]][
                "encoding"
            ].values,
            axis=1,
        ).T
        return euclidean(
            np.average(encodings_i, axis=0), np.average(encodings_j, axis=0)
        )

    def sample_morgan_tanimoto(
        self,
        n: int,
        seed: int = 42,
    ):
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3)
        self.fingerprints = [fpgen.GetFingerprint(x) for x in self.sdf]
        mol_ids = self.dataset["mol_idx"].unique()
        self.id_mapping = dict(enumerate(mol_ids))
        picker = MaxMinPicker()
        self.train_ids = list(
            map(
                self.id_mapping.get,
                list(
                    picker.LazyBitVectorPick(
                        self.fingerprints, len(self.fingerprints), n, seed=seed
                    )
                ),
            )
        )

    def sample_atomic_encodings(
        self,
        distance: Callable,
    ):
        def _sample_atomic_encodings(
            n: int,
            seed: int = 42,
        ):
            mol_ids = self.dataset["mol_idx"].unique()
            self.id_mapping = dict(enumerate(mol_ids))
            picker = MaxMinPicker()
            self.train_ids = list(
                map(
                    self.id_mapping.get,
                    list(picker.LazyPick(distance, len(mol_ids), n, seed=seed)),
                )
            )

        return _sample_atomic_encodings

    def sample_dataset(self, based_on: str, n: int, seed: int = 42, suffix: str = ""):
        if based_on == "morgan":
            sampler = self.sample_morgan_tanimoto
        elif based_on == "random":
            sampler = self.sample_random
        elif based_on == "hausdorff_encodings":
            sampler = self.sample_atomic_encodings(self.distance_hausdorff_encodings)
        elif based_on == "euclidian_mean_encodings":
            sampler = self.sample_atomic_encodings(
                self.distance_euclidian_mean_encodings
            )
        else:
            raise NotImplementedError(
                "You can sample based on: 'morgan', 'hausdorff_encodings' or 'random"
            )
        sampler(n=n, seed=seed)
        self.save_dataset(f"{based_on}_{seed}/{n}_{suffix}")

    def sample_random(self, n: int = 100, seed: int = 42):
        rng = np.random.default_rng(seed=seed)
        indices = list(
            rng.choice(list(self.dataset["mol_idx"].unique()), size=n, replace=False)
        )
        self.train_ids = indices
