#preprocessing utilities
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem as Chem

SPECTRUM_LABELS = {
    "C": "C13",
    "H": "1H",
}


def read_shifts(molecule, spectrum_type="C"):
    spectrum_label = SPECTRUM_LABELS[spectrum_type]
    raw_string = molecule.GetProp(f"{spectrum_label} Chemical Shift")
    spectrum = []
    dict_spectrum = {}
    for entry in raw_string.split("|")[:-1]:
        index, shift = entry.split(",")
        dict_spectrum[int(index)] = float(shift)
    for i, atom in enumerate(molecule.GetAtoms()):
        try:
            spectrum.append(dict_spectrum.get(i))
        except KeyError:
            spectrum.append(np.nan)
    return np.array(spectrum, dtype=np.float32)


def parse_shfits(suppplier, spectrum_type="C"):
    mol_idxs = []
    atom_idxs = []
    shifts = []
    for mol_idx, mol in enumerate(suppplier):
        shift_list = read_shifts(mol, spectrum_type=spectrum_type)
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            shift = shift_list[atom_idx]
            if not np.isnan(shift):
                mol_idxs.append(mol_idx)
                atom_idxs.append(atom_idx)
                shifts.append(shift)
    return pd.DataFrame(
        {
            "mol_idx": mol_idxs,
            "atom_idx": atom_idxs,
            "shift": shifts,
        }
    ).astype({"atom_idx": np.int32, "mol_idx": np.int32, "shift": np.float32})


def find_lowest_conformer(mol):

    m = Chem.AddHs(mol)
    # Number of conformers to be generated
    num_of_conformer = 1000
    max_iter = 500
    # Default values for min energy conformer
    min_energy_MMFF = 10000
    min_energy_index_MMFF = 0

    method = Chem.ETKDGv2()
    method.numThreads = 12
    # Generate conformers (stored in side the mol object)
    # cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=num_of_conformer)
    try:
        cids = Chem.EmbedMultipleConfs(m, numConfs=num_of_conformer, params=method)
        results = Chem.MMFFOptimizeMoleculeConfs(
            m, maxIters=max_iter, mmffVariant="MMFF94s", numThreads=12
        )
    except:
        return None

    # Search for the min energy conformer from results(tuple(is_converged,energy))

    # print("\nSearching conformers by MMFF ")
    for index, result in enumerate(results):
        if min_energy_MMFF > result[1]:
            min_energy_MMFF = result[1]
            min_energy_index_MMFF = index
            # print(min_energy_index_MMFF,":",min_energy_MMFF)
    try:
        output = Chem.Mol(m, False, min_energy_index_MMFF)
    except:
        output = None

    return output
