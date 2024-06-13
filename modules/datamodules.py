import numpy as np
from lightning.pytorch import LightningDataModule
from torch.utils.data import random_split, Dataset, DataLoader, default_collate
import torch
from torch.utils.data import default_collate
import pandas as pd
from rdkit.Chem.AllChem import SDMolSupplier
from preprocessing import encoders
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader as pygDataLoader
from preprocessing.encoders import UniMolEncoder, MACEEncoder, MoltoGraphEncoder


def collate(batch):
    return {
        key: default_collate([entry[key] for entry in batch]) for key in batch[0].keys()
    }


class MolecularShiftsDataset(InMemoryDataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        encoding: str,
        sdf: SDMolSupplier,
        name: str = "data",
        root: str = "data/graph_datasets/",  # PUT THE DIRECTORY WHERE YOU WANT GRAPH DATASETS
        transform=None,
        pre_transform=None,
    ):
        self.encoding = encoding
        self.name = name
        self.dataset = dataset
        self.sdf = sdf
        super().__init__(root, transform, pre_transform)
        self.load(f"{self.root}/processed/{self.name}_{self.encoding}.pt")

    @property
    def processed_file_names(self):
        return [f"{self.name}_{self.encoding}.pt"]

    def process(self):
        encoder = encoders.MoltoGraphEncoder()
        # Read data into huge `Data` list.
        data_list = encoder.encode(self.sdf, self.dataset)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


class AtomicShiftsDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe.to_dict("records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            k: torch.tensor(self.data[idx][k]) for k in self.data[idx] if k != "shift"
        }
        item["shift"] = torch.tensor([self.data[idx].get("shift", np.nan)])
        return item


class MolecularShiftsDatamodule(LightningDataModule):
    """
    train_data: name of parquet file inside data/
    test_data: name of parquet file inside data/
    sdf_train: sdf file with molecules used to generate train data
    batch size: batch size (default: 1)
    encoding: name of encoding to use
    validation: (bool) split the dataset to validate
    val_ratio: the ratio of validation split
    predict: path to sdf for molecule(s) to predict
    """

    def __init__(
        self,
        train_data: str = None,
        test_data: str = None,
        sdf_train: str = None,
        batch_size: int = 1,
        encoding: str = None,
        val_ratio: float = 0.1,
        validation: bool = True,
        predict: str = None,
        num_workers: int = 7,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.encoding = encoding
        self.train_data = train_data
        self.sdf_train = sdf_train
        self.test_data = test_data
        self.validation = validation
        self.predict = predict
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.save_hyperparameters(
            ignore=["predict", "validation", "num_workers", "val_ratio"]
        )

    def setup(self, stage: str):
        if stage == "fit":
            df_train = pd.read_parquet(
                f"data/atomic_datasets/{self.train_data}_{self.encoding}.parquet"  # CHANGE THE BEGINNING TO YOUR ATOMIC DATASET DIRECTORY
            )
            if not self.sdf_train:
                self.sdf_train = self.train_data
            sdf_train = SDMolSupplier(
                f"raw_data/{self.sdf_train}.sdf",  # CHANGE THE BEGINNING TO YOUR SDF FILES DIRECTORY
                True,
                False,
                True,
            )
            self.train_data = MolecularShiftsDataset(
                df_train,
                encoding=self.encoding,
                sdf=sdf_train,
                name=self.train_data,
            )
            if self.validation:
                self.train_data, self.val_data = random_split(
                    self.train_data,
                    [1 - self.val_ratio, self.val_ratio],
                    generator=torch.Generator().manual_seed(42),
                )

        if stage == "test":
            if self.test_data == "training":
                df_train = pd.read_parquet(
                    f"data/atomic_datasets/{self.train_data}_{self.encoding}.parquet"  # CHANGE THE BEGINNING TO YOUR ATOMIC DATASET DIRECTORY
                )
                if not self.sdf_train:
                    self.sdf_train = self.train_data
                sdf_train = SDMolSupplier(
                    f"raw_data/{self.sdf_train}.sdf",  # CHANGE THE BEGINNING TO YOUR SDF FILES DIRECTORY
                    True,
                    False,
                    True,
                )
                self.test_data = MolecularShiftsDataset(
                    df_train,
                    encoding=self.encoding,
                    sdf=sdf_train,
                    name=self.train_data,
                )
            else:
                df_test = pd.read_parquet(
                    f"data/atomic_datasets/{self.test_data}_{self.encoding}.parquet"  # CHANGE THE BEGINNING TO YOUR ATOMIC DATASET DIRECTORY
                )
                sdf_test = SDMolSupplier(
                    f"raw_data/{self.test_data}.sdf",
                    True,
                    False,
                    True,  # CHANGE THE BEGINNING TO YOUR SDF FILES DIRECTORY
                )
                self.test_data = MolecularShiftsDataset(
                    df_test,
                    encoding=self.encoding,
                    sdf=sdf_test,
                    name=self.test_data,
                )
        if stage == "predict":
            if type(self.predict) == str:
                self.predict = SDMolSupplier(self.predict, True, False, True)
            elif type(self.predict) == SDMolSupplier:
                pass
            elif type(self.predict) == list:
                pass
            else:
                raise NotImplementedError(
                    "Currently only path to SDF file or SDMolSupplier are supported"
                )
            if self.encoding == "mace_l":
                encoder = MACEEncoder()
            elif self.encoding == "unimol":
                encoder = UniMolEncoder()
            else:
                raise NotImplementedError(
                    "Currently only mace_l and unimol encoders are supported"
                )
            df = encoder.encode(self.predict)
            df["shift"] = np.nan
            encoder = MoltoGraphEncoder()
            self.predict_data = encoder.encode(self.predict, df)

    def train_dataloader(self):
        return pygDataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return pygDataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return pygDataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return pygDataLoader(
            self.predict_data,
            num_workers=7,
            batch_size=128,
        )


class AtomicShiftsDatamodule(LightningDataModule):
    """
    train_data: name of parquet dataframe with encodings
    test_data: name of parquet datagrame with encodings
    batch size: batch size (default: 1)
    encoding: name of encoding to use
    validation: (bool) split the dataset to validate
    predict: path to sdf or xyz file for molecule to predict
    """

    def __init__(
        self,
        train_data: str = None,
        test_data: str = None,
        predict: str = None,
        batch_size: int = 1,
        encoding: str = None,
        validation: bool = True,
        val_split: str = "by_atom",
        val_ratio: float = 0.1,
        num_workers: int = 7,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.encoding = encoding
        self.train_data = train_data
        self.val_split = val_split
        self.test_data = test_data
        self.predict = predict
        self.validation = validation
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.save_hyperparameters(
            ignore=["predict", "validation", "num_workers", "val_ratio"]
        )

    def setup(self, stage: str):
        self.df_train = pd.read_parquet(
            f"data/atomic_datasets/{self.train_data}_{self.encoding}_C.parquet"  # CHANGE THE BEGINNING TO YOUR ATOMIC DATASET DIRECTORY
        )
        if stage == "fit":
            self.train_data = AtomicShiftsDataset(self.df_train)
            if self.validation:
                if self.val_split == "by_atom":
                    self.train_data, self.val_data = random_split(
                        self.train_data,
                        [1 - self.val_ratio, self.val_ratio],
                        generator=torch.Generator().manual_seed(42),
                    )
                elif self.val_split == "by_molecule":
                    mol_ids = self.df_train["mol_idx"].unique()
                    train_ids = (
                        torch.multinomial(
                            torch.ones(len(mol_ids)),
                            num_samples=int(
                                np.round(len(mol_ids) * (1 - self.val_ratio))
                            ),
                            replacement=False,
                            generator=torch.Generator().manual_seed(42),
                        )
                        .cpu()
                        .numpy()
                    )
                    train_ids = mol_ids[train_ids]
                    self.train_data = AtomicShiftsDataset(
                        self.df_train[self.df_train["mol_idx"].isin(train_ids)]
                    )
                    self.val_data = AtomicShiftsDataset(
                        self.df_train[~self.df_train["mol_idx"].isin(train_ids)]
                    )
        if stage == "test":
            if self.test_data == "training":
                self.df_train = pd.read_parquet(
                    f"data/atomic_datasets/{self.train_data}_{self.encoding}_C.parquet"  # CHANGE THE BEGINNING TO YOUR ATOMIC DATASET DIRECTORY
                )
                self.test_data = AtomicShiftsDataset(self.df_train)
            else:
                self.df_test = pd.read_parquet(
                    f"data/atomic_datasets/{self.test_data}_{self.encoding}_C.parquet"  # CHANGE THE BEGINNING TO YOUR ATOMIC DATASET DIRECTORY
                )
                self.test_data = AtomicShiftsDataset(self.df_test)

        if stage == "predict":
            if self.encoding == "unimol":
                encoder = encoders.UniMolEncoder()
            elif self.encoding == "mace_l":
                encoder = encoders.MACEEncoder(model_size="large")
            elif self.encoding:
                raise NotImplementedError(f"{self.encoding} encoding not implemented.")
            else:
                raise ValueError("You have to specify encoding model uses.")
            if self.predict.endswith(".xyz"):
                # TO DO
                pass
            if isinstance(self.predict, SDMolSupplier):
                supp = list(self.predict)
            elif type(self.predict) == list:
                pass
            else:
                supp = SDMolSupplier(self.predict, True, False, True)
            encodings = encoder.encode(supp)
            for mol_id, mol in enumerate(supp):
                for atom_id, atom in enumerate(mol.GetAtoms()):
                    if atom.GetSymbol() != "C":
                        encodings.drop(
                            encodings.loc[
                                (encodings["mol_idx"] == mol_id)
                                & (encodings["atom_idx"] == atom_id)
                            ].index,
                            inplace=True,
                        )

            self.predict_data = AtomicShiftsDataset(encodings)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            collate_fn=collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            collate_fn=collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            collate_fn=collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            collate_fn=collate,
            num_workers=7,
            batch_size=2048,
        )
