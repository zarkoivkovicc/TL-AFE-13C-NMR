"This script is used to transform the dataset from 10.1021/acs.jcim.0c00195 to sdf file so it can be used for this work."
import pandas as pd
from rdkit import Chem


def write_shifts(shift_dict):
    shifts = ""
    for k in shift_dict:
        shifts += f"{k},{shift_dict[k] :.2f}|"
    return shifts


data = pd.read_pickle("NMR_data_13C.pickle") # put the direction to pickle file from the paper here
train = data["train_df"]
mols_train = train["rdmol"]
spectra_train = train["value"]
molecules = []
for mol_id, mol in mols_train.items():
    try:
        Chem.SanitizeMol(mol)
    except:
        continue
    mol.SetProp("SMILES", Chem.MolToSmiles(mol))
    mol.SetProp("_Name", str(mol_id))
    shifts = dict(sorted(spectra_train[mol_id][0].items()))
    mol.SetProp("C13 Chemical Shift", write_shifts(shifts))
    molecules.append(mol)
with Chem.SDWriter("raw_data/NMR_FF_train.sdf") as w:
    for m in molecules:
        w.write(m)

test = data["test_df"]
mols_test = test["rdmol"]
spectra_test = test["value"]
molecules = []
for mol_id, mol in mols_test.items():
    try:
        Chem.SanitizeMol(mol)
        Chem.AssignAtomChiralTagsFromStructure(mol)
        Chem.AssignStereochemistry(mol)
    except:
        continue
    mol.SetProp("SMILES", Chem.MolToSmiles(mol))
    mol.SetProp("_Name", str(mol_id))
    shifts = dict(sorted(spectra_test[mol_id][0].items()))
    mol.SetProp("C13 Chemical Shift", write_shifts(shifts))
    molecules.append(mol)
with Chem.SDWriter("raw_data/NMR_FF_test.sdf") as w:
    for m in molecules:
        w.write(m)
