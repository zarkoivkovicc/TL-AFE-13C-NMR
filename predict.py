# Author: Zarko Ivkovic
# NOTE: This is not optimized, it can be slow for a lot of molecules
from modules.models import SimpleGNN
from modules.datamodules import MolecularShiftsDatamodule
from lightning.pytorch import Trainer
import pandas as pd
from argparse import ArgumentParser
from preprocessing.preprocess import find_lowest_conformer
from rdkit.Chem import SDMolSupplier, MolToSmiles

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Preprocess data and save encodings or shifts as dictionaries using numpy"
    )
    parser.add_argument("sdf", type=str, help="SDF file to process")
    parser.add_argument(
        "--SMILES", type=str, help="Smiles string of the molecule you want to predict"
    )
    args = parser.parse_args()
    supp = SDMolSupplier(args.sdf, True, False, True)
    embeded = [find_lowest_conformer(mol) for mol in supp]
    for mol in embeded:
        mol.SetProp("SMILES", MolToSmiles(mol))
    print("Succesfully embeded molecules!")
    model1 = SimpleGNN.load_from_checkpoint("models/mace_gnn/model.pt")
    model2 = SimpleGNN.load_from_checkpoint("models/unimol_gnn/model.pt")
    data1 = MolecularShiftsDatamodule(
        train_data="NMR_FF_train",
        test_data="NMR_FF_test",
        batch_size=128,
        encoding="mace_l",
        predict=embeded,
    )
    data2 = MolecularShiftsDatamodule(
        train_data="NMR_FF_train",
        test_data="NMR_FF_test",
        batch_size=128,
        encoding="unimol",
        predict=embeded,
    )
    trainer1 = Trainer(devices=1)
    results1 = trainer1.predict(model1, data1)
    trainer2 = Trainer(devices=1)
    results2 = trainer2.predict(model2, data2)
    results = pd.merge(
        left=trainer1.model.results,
        right=trainer2.model.results,
        how="inner",
        on=["mol_idx", "atom_idx"],
        suffixes=["_mace", "_unimol"],
        validate="one_to_one",
    )
    results["predicted_shift"] = (
        results["predicted_shift_mace"] + results["predicted_shift_unimol"]
    ) / 2
    results.loc[:, ["mol_idx", "atom_idx", "predicted_shift"]].to_csv(f"predict.csv")
