from rdkit.Chem import AllChem as Chem
import pandas as pd


def get_shift_dict(molecule):
    raw_string = molecule.GetProp(f"C13 chemical shift")
    dict_spectrum = {}
    for entry in raw_string.split("|")[:-1]:
        index, shift = entry.split(",")
        dict_spectrum[int(index)] = float(shift)
    return dict_spectrum


def extract_shift_dict(text):
    shift_dict = {}
    for shift in text.split("|")[:-1]:
        entry = shift.split(";")
        shift_dict[int(entry[2])] = float(entry[0])
    return shift_dict


def clean_shifts(mol):
    shifts = []
    n_shifts = []
    for i in range(5):
        try:
            dict_shift = extract_shift_dict(mol.GetProp(f"Spectrum 13C {i}"))
            shifts.append(dict_shift)
            n_shifts.append(len(dict_shift))
        except:
            pass
    for i, _ in enumerate(shifts):
        if n_shifts[i] != max(n_shifts):
            shifts.pop(i)
    if len(shifts) >= 1:
        averaged_shifts = {k: 0 for k in shifts[0]}
    else:
        return None
    for k in averaged_shifts:
        for spec in shifts:
            averaged_shifts[k] += spec[k]
    averaged_shifts = {k: averaged_shifts[k] / len(shifts) for k in averaged_shifts}
    averaged_shifts = dict(sorted(averaged_shifts.items()))
    return averaged_shifts


DFT = list(
    Chem.SDMolSupplier("../raw_data/CASCADE_datasets/DFT8K_DFT.sdf", True, False, True)
)
EXP = list(
    Chem.SDMolSupplier("../raw_data/CASCADE_datasets/NMR8K.sdf", False, True, True)
)

mol_ids = []
atom_ids = []
exp_shift = []
dft_shift = []
for mol1 in EXP:
    exp_shifts = clean_shifts(mol1)
    for index, atom in enumerate(mol1.GetAtoms()):
        atom.SetAtomMapNum(index + 1)  # Because 0 means no atom mapping
    mol_1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol1))
    Chem.SanitizeMol(mol_1)
    mol_1 = Chem.AddHs(mol_1)
    atoms = mol_1.GetAtoms()
    mapping = {atom.GetAtomMapNum() - 1: k for k, atom in enumerate(atoms)}
    for index, atom in enumerate(mol_1.GetAtoms()):
        atom.SetAtomMapNum(0)
    smiles1 = Chem.MolToSmiles(mol_1, isomericSmiles=False)
    order1 = [int(i) for i in mol_1.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")]
    if not exp_shifts:
        continue
    for mol2 in DFT:
        if smiles1 == Chem.MolToSmiles(mol2, isomericSmiles=False):
            try:
                dft_shifts = get_shift_dict(mol2)
            except:
                break
            print(mol2.GetProp("_Name"))
            order2 = [
                int(i) for i in mol2.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")
            ]
            mapping2 = dict(zip(order1, order2))
            for k in exp_shifts:
                mol_ids.append(int(mol2.GetProp("_Name")))
                atom_ids.append(mapping2[mapping[k]])
                exp_shift.append(exp_shifts[k])
                dft_shift.append(dft_shifts[mapping2[mapping[k]]])

results = pd.DataFrame(
    {
        "mol_idx": mol_ids,
        "atom_idx": atom_ids,
        "experimental_shift": exp_shift,
        "dft_shift": dft_shift,
    }
)
results.to_csv("dft_vs_exp_correct_DFT_geom.csv")
