import numpy as np
from argparse import ArgumentParser
from rdkit.Chem import AllChem as Chem
from encoders import UniMolEncoder, MACEEncoder, RDKitEnocder
import pandas as pd
from preprocess import parse_shfits

MACE_LABELS = {
    "mace_s": "small",
    "mace_m": "medium",
    "mace_l": "large",
}

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Preprocess data and save encodings or shifts as dictionaries using numpy"
    )
    parser.add_argument("sdf", type=str, help="SDF file to process")
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["unimol", "mace_s", "mace_m", "mace_l"],
        help="choose atomic encodings",
        required=True,
    )
    parser.add_argument(
        "--rdkit_encodings",
        action="store_true",
        help="if true adds rdkit encodings column to the dataframe",
    )
    parser.add_argument(
        "--onlyC",
        action="store_true",
        help="store additional dataframe with only labeled carbon atoms",
    )
    parser.add_argument(
        "--mace_layers",
        type=int,
        choices=[1, 2],
        help="choose layer to extract encoding from",
        default=1,
    )

    args = parser.parse_args()

    supplier = Chem.SDMolSupplier(
        f"../raw_data/{args.sdf}.sdf",
        sanitize=True,
        removeHs=False,
        strictParsing=True,
    )
    if args.encoder == "unimol":
        encoder = UniMolEncoder()
    elif args.encoder[:-1] == "mace_":
        encoder = MACEEncoder(
            model_size=MACE_LABELS[args.encoder], layer=args.mace_layers
        )
    else:
        raise NotImplementedError(f"{args.encoder} isn't implemented.")
    encodings = encoder.encode(supplier)
    shifts = parse_shfits(supplier, spectrum_type="C")
    if args.rdkit_encodings:
        supplier = Chem.SDMolSupplier(
            f"../raw_data/{args.sdf}.sdf",
            sanitize=True,
            removeHs=False,
            strictParsing=True,
        )
        rdkit_encoder = RDKitEnocder()
        rdkit_encodings = rdkit_encoder.encode(supplier)
        encodings = pd.merge(
            encodings,
            rdkit_encodings,
            how="left",
            on=["mol_idx", "atom_idx"],
            suffixes=[None, "_rdkit"],
        )
    data = pd.merge(
        encodings,
        shifts,
        how="left",
        on=["mol_idx", "atom_idx"],
    )
    encoder_name = args.encoder
    if args.mace_layers != 1:
        encoder_name = f"{args.encoder}{args.mace_layers}"
    data.to_parquet(f"../data/{args.sdf}_{encoder_name}.parquet")
    if args.onlyC:
        data.dropna().to_parquet(f"../data/{args.sdf}_{encoder_name}_C.parquet")
