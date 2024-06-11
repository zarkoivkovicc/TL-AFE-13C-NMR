import numpy as np
import torch
from unimol_tools import UniMolRepr
from rdkit.Chem.AllChem import SDMolSupplier, MolToMolBlock
from rdkit import Chem
from ase.io import read as readase
from mace.calculators import MACECalculator
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import io
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data


class Encoder(ABC):
    def __init__(self):
        self.mol_idxs = []
        self.atom_idxs = []
        self.encodings = []

    @abstractmethod
    def encode(self, input) -> pd.DataFrame:
        pass

    def make_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "mol_idx": self.mol_idxs,
                "atom_idx": self.atom_idxs,
                "encoding": self.encodings,
            }
        ).astype({"atom_idx": np.int32, "mol_idx": np.int32, "encoding": object})


class UniMolEncoder(Encoder):
    def __init__(self):
        super().__init__()
        self.encoder = UniMolRepr(data_type="molecule", remove_hs=False)

    def encode(self, input: SDMolSupplier | list[str]) -> pd.DataFrame:
        "Takes list of molecules or smiles strings and returns the encodings."
        if isinstance(input, str):
            raise NotImplementedError(
                "SMILES isn't currently unimplemented for UniMol encoder"
            )
        else:
            supplier = input
            elements = []
            coordinates = []
            for mol_idx, mol in enumerate(supplier):
                # potentially parallelize this for loop
                coordinates.append(
                    np.array(mol.GetConformer().GetPositions(), dtype=np.float32)
                )
                atoms = mol.GetAtoms()
                elements.append([atom.GetSymbol() for atom in atoms])
                self.mol_idxs.extend([mol_idx] * len(atoms))
                self.atom_idxs.extend([id for id, _ in enumerate(atoms)])
            entry = {"atoms": elements, "coordinates": coordinates}
            unimol_repr = self.encoder.get_repr(entry, return_atomic_reprs=True)
            for mol_idx, atom_idx in zip(self.mol_idxs, self.atom_idxs):
                self.encodings.append(
                    np.array(
                        unimol_repr["atomic_reprs"][mol_idx][atom_idx], dtype=np.float32
                    )
                )
            return self.make_dataframe()


class MACEEncoder(Encoder):
    def __init__(self, model_size: str = "large", layer: int = 1):
        super().__init__()
        self.encoder = MACECalculator(
            model_paths=f"/home/zarko/mace_off23/MACE-OFF23_{model_size}.model",
            device="cuda" if torch.cuda.is_available() else "cpu",
            default_dtype="float64",
        )
        self.layer = layer

    def encode(self, input) -> pd.DataFrame:
        if type(input) == str:
            raise NotImplementedError(
                "SMILES isn't currently unimplemented for MACE encoder"
            )
        else:
            supplier = input
            for mol_idx, mol in enumerate(supplier):
                # potentially parallelize this for loop
                f = io.StringIO(MolToMolBlock(mol))
                descriptors = self.encoder.get_descriptors(
                    readase(f, format="sdf"), num_layers=self.layer
                )
                if self.layer == 2:
                    descriptors = descriptors[
                        :, : int(descriptors.shape[1] / self.layer)
                    ]
                for atom_idx, atom in enumerate(mol.GetAtoms()):
                    self.mol_idxs.append(mol_idx)
                    self.atom_idxs.append(atom_idx)
                    self.encodings.append(
                        np.array(descriptors[atom_idx], dtype=np.float32)
                    )
            return self.make_dataframe()


class RDKitEnocder(Encoder):
    def __init__(self):
        super().__init__()
        self.chirality_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", dtype=np.int32
        )
        self.chirality_encoder.fit([["R"], ["S"]])
        self.hybridization_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", dtype=np.int32
        )
        self.hybridization_encoder.fit(
            [["S"], ["SP"], ["SP2"], ["SP3"], ["SP3D"], ["SP3D2"]]
        )

    def encode_chirality(self, x):
        try:
            res = x.GetProp("_CIPCode")
        except:
            res = None
        return self.chirality_encoder.transform([[res]]).squeeze()

    def encode_hybridization(self, x):
        return self.hybridization_encoder.transform(
            [[str(x.GetHybridization())]]
        ).squeeze()

    def encode_atom(self, atom):
        is_aromatic = [int(atom.GetIsAromatic())]
        formal_charge = [int(atom.GetFormalCharge())]
        num_hydrogens = [int(atom.GetTotalNumHs(includeNeighbors=True))]
        num_carbons = [
            list(map(lambda x: x.GetSymbol(), atom.GetNeighbors())).count("C")
        ]
        num_hetero = [atom.GetTotalDegree() - num_hydrogens[0] - num_carbons[0]]
        ring_info = [int(atom.IsInRingSize(i)) for i in range(3, 9)]
        chirality_info = self.encode_chirality(atom)
        hybridization_info = self.encode_hybridization(atom)
        return np.concatenate(
            [
                is_aromatic,
                formal_charge,
                num_hydrogens,
                num_carbons,
                num_hetero,
                ring_info,
                chirality_info,
                hybridization_info,
            ],
            dtype=np.float32,
        )

    def encode(self, input) -> pd.DataFrame:
        if type(input) == str or type(input) == list:
            raise NotImplementedError(
                "SMILES input or xyz currently unimplemented for UniMol encoder"
            )
        else:
            supplier = input
            for mol_idx, mol in enumerate(supplier):
                for atom_idx, atom in enumerate(mol.GetAtoms()):
                    self.mol_idxs.append(mol_idx)
                    self.atom_idxs.append(atom_idx)
                    self.encodings.append(self.encode_atom(atom))
            return self.make_dataframe()


class MoltoGraphEncoder(Encoder):
    def __init__(self) -> None:
        self.bond_type_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", dtype=np.float32
        )
        self.bond_type_encoder.fit([["SINGLE"], ["DOUBLE"], ["TRIPLE"], ["AROMATIC"]])

    def encode_bond(self, x):
        bond_type = self.bond_type_encoder.transform([[str(x.GetBondType())]]).squeeze()
        bond_conjg = int(x.GetIsConjugated())
        bond_ring = int(x.IsInRing())
        return np.concatenate(
            [bond_type, [bond_conjg], [bond_ring]],
            dtype=np.float32,
        )

    def mol_to_graph(self, mol: Chem.Mol, mol_id: int):
        has_shifts = True
        try:
            mol.GetProp("C13 Chemical Shift")
        except:
            has_shifts = False
        xs = []
        ys = []
        masks = []
        atom_ids = []
        for atom_id, atom in enumerate(mol.GetAtoms()):
            example = self.encodings[
                (self.encodings["mol_idx"] == mol_id)
                & (self.encodings["atom_idx"] == atom_id)
            ]
            xs.append(example["encoding"].item())
            ys.append(example["shift"].item())
            if has_shifts:
                masks.append(not np.isnan(example["shift"].item()))
            else:
                masks.append(atom.GetSymbol() == "C")

            atom_ids.append(atom_id)
        x = torch.tensor(np.array(xs), dtype=torch.float32)
        y = torch.tensor(np.array(ys), dtype=torch.float32).view(-1, 1)
        mask = torch.tensor(np.array(masks), dtype=torch.bool)
        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_indices += [[i, j], [j, i]]
            e = self.encode_bond(bond)
            edge_attrs += [e, e]

        edge_index = torch.tensor(edge_indices)
        edge_index = edge_index.t().to(torch.int64).view(2, -1)
        edge_attrs = torch.tensor(np.array(edge_attrs), dtype=torch.float32)

        if edge_index.numel() > 0:  # Sort indices.
            perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
            edge_index, edge_attrs = edge_index[:, perm], edge_attrs[perm]
        mol_id = torch.tensor(mol_id, dtype=torch.int32).expand(y.shape[0])
        atom_id = torch.tensor(atom_ids, dtype=torch.int32)
        return Data(
            x=x,
            y=y,
            mask=mask,
            edge_index=edge_index,
            edge_attr=edge_attrs,
            smiles=mol.GetProp("SMILES"),
            mol_id=mol_id,
            atom_id=atom_id,
        )

    def encode(self, sdf: SDMolSupplier, encodings: pd.DataFrame):
        mol_ids = encodings["mol_idx"].unique()
        sdf = list(sdf)
        self.sdf = [sdf[int(mol_id)] for mol_id in mol_ids][::-1]
        self.encodings = encodings
        self.graph_data = []
        for mol_id in mol_ids:
            # potentially paralelize this for loop
            self.graph_data.append(self.mol_to_graph(self.sdf.pop(), mol_id))
        return self.graph_data
