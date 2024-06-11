import pandas as pd
import seaborn as sns
import math
import numpy as np
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw
from rdkit.Chem.AllChem import SDMolSupplier
from rdkit.Chem import rdDepictor, AssignStereochemistryFrom3D
from rdkit import Chem
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import linregress
from rdkit.Chem.Draw import DrawingOptions
import io
import cairosvg
from IPython.display import SVG

cm = 1 / 2.54  # centimeters in inches
rdDepictor.SetPreferCoordGen(True)
DrawingOptions.dotsPerAngstrom = 100


def annotate(data, **kws):
    r, p = pearsonr(data["predicted_shift"], data["true_shift"])
    mae = mean_absolute_error(data["predicted_shift"], data["true_shift"])
    rmse = np.sqrt(mean_squared_error(data["predicted_shift"], data["true_shift"]))
    ax = plt.gca()
    ax.text(0.15, 0.8, f"$\\rho$: {r : .4f}", transform=ax.transAxes)
    ax.text(0.15, 0.75, f"MAE: { mae: .2f} ppm", transform=ax.transAxes)
    ax.text(0.15, 0.70, f"RMSE: { rmse: .2f} ppm", transform=ax.transAxes)


class ResultVisualizer(object):
    def __init__(
        self, sdf: SDMolSupplier, results: pd.DataFrame, style="white", context="paper"
    ) -> None:
        self.sdf = list(sdf)
        self.results = results
        sns.set_style(style)
        sns.set_context(context)

    def visualize_mols(
        self,
        ids: list[int],
        mols_per_row: int = 5,
        size: int = 200,
        min_font_size: int = 12,
        file_name: str | None = None,
        lower_error_bound: float = 1.5,
        higher_error_bound: float = 4.5,
        width: int | None = None,
        height: int | None = None,
    ):
        def get_color(error):
            def get_power(error, lower, higher):
                if error < lower:
                    return 0.0
                elif error < higher:
                    return (error - lower) / (higher - lower)
                else:
                    return 1.0

            power = get_power(abs(error), lower_error_bound, higher_error_bound)
            blue = 0.0
            if 0 <= power < 0.5:  # first, green stays at 100%, red raises to 100%
                green = 1.0
                red = 2 * power
            elif 0.5 <= power <= 1:  # then red stays at 100%, green decays
                red = 1.0
                green = 1.0 - 2 * (power - 0.5)

            return (red, green, blue)

        mols_per_row = min(len(ids), mols_per_row)
        rows = math.ceil(len(ids) / mols_per_row)
        mols = []
        atom_ids = []
        atom_colors = []
        for id in ids:
            mols.append(self.sdf[id])
            atom_ids.append(
                list(self.results[self.results["mol_idx"] == id]["atom_idx"])
            )
            atom_colors.append({})
            for atom_id in list(
                self.results[self.results["mol_idx"] == id]["atom_idx"]
            ):
                entry = self.results[
                    (
                        (self.results["mol_idx"] == id)
                        & (self.results["atom_idx"] == atom_id)
                    )
                ]
                error = entry["error"].item()
                predicted = entry["predicted_shift"].item()
                true = entry["true_shift"].item()
                atom_colors[-1][atom_id] = get_color(error)
                mols[-1].GetAtoms()[atom_id].SetProp("atomNote", f"{error:+.2f}")

        molecules = []
        for mol_id, mol in enumerate(mols):
            # AssignStereochemistryFrom3D(mol)
            for atom_id, atom in enumerate(mol.GetAtoms()):
                atom.SetAtomMapNum(atom_id)
            mol = Chem.RemoveHs(mol)
            rdDepictor.Compute2DCoords(mol)
            rdDepictor.StraightenDepiction(mol)
            # mapping is done here because removing hydrogens
            # changes order of atoms in some cases
            mapping = {}
            for atom_id, atom in enumerate(mol.GetAtoms()):
                mapping[atom.GetAtomMapNum()] = atom_id
                atom.SetAtomMapNum(0)
            molecules.append(mol)
            atom_ids[mol_id] = [*map(mapping.get, atom_ids[mol_id])]

        d2d = rdMolDraw2D.MolDraw2DSVG(mols_per_row * size, rows * size, size, size)
        d2d.drawOptions().minFontSize = min_font_size
        d2d.drawOptions().legendFontSize = min_font_size
        # d2d.drawOptions().addStereoAnnotation = True
        # d2d.drawOptions().addAtomIndices = True
        d2d.DrawMolecules(
            molecules,
            legends=[str(i) for i in ids],
            highlightAtoms=atom_ids,
            highlightBonds=[None for i in ids],
            highlightAtomColors=atom_colors,
        )
        d2d.FinishDrawing()
        if file_name:
            cairosvg.svg2pdf(
                bytestring=d2d.GetDrawingText(),
                write_to=file_name,
                dpi=800,
            )
        return SVG(d2d.GetDrawingText())

    def visualize_error_dist(
        self,
        file_name: str | None = None,
        lim: float | None = None,
        size: tuple[int] | None = (8.3 * cm, 8.3 * cm),
    ):
        plot = sns.displot(self.results, x="error")
        plot.set_xlabels("Error [ppm]")
        if lim:
            plot.set(xlim=(-lim, +lim))
        if file_name:
            plot.figure.set_size_inches(*size)
            plot.tight_layout()
            plot.savefig(file_name, dpi=800)
        return plot

    def visualize_true_vs_predicted(
        self,
        file_name: str | None = None,
        size: tuple[int] | None = (8.3 * cm, 8.3 * cm),
    ):
        plot = sns.lmplot(
            self.results,
            x="predicted_shift",
            y="true_shift",
            line_kws={"color": "black", "linestyle": "dashed", "linewidth": 1},
            scatter_kws={"s": 0.1},
        )
        plot.set_xlabels("Predicted shift [ppm]")
        plot.set_ylabels("True shift [ppm]")
        plot.map_dataframe(annotate)
        if file_name:
            plot.figure.set_size_inches(*size)
            plot.tight_layout()
            plot.savefig(file_name, dpi=800)
        return plot

    def visualize_residuals(self, file_name=None):
        residuals = (
            self.results.loc[:, "predicted_shift"] - self.results.loc[:, "true_shift"]
        )
        line = linregress(self.results.loc[:, "predicted_shift"], residuals)
        plot = sns.scatterplot(x=self.results.loc[:, "predicted_shift"], y=residuals)
        plot.axhline(color="black", linestyle="dashed")
        x_min = min(self.results.loc[:, "predicted_shift"])
        x_max = max(self.results.loc[:, "predicted_shift"])
        y_min = line.intercept + line.slope * x_min
        y_max = line.intercept + line.slope * x_max
        plot.plot([x_min, x_max], [y_min, y_max], color="red")
        plot.set_xlabel("Predicted shift [ppm]")
        plot.set_ylabel("Residual [ppm]")
        if file_name:
            plot.savefig(file_name, dpi=800)
        return plot

    def visualize_outliers(self, min_error: float = 5.0, **kwargs):
        outliers = self.results[abs(self.results["error"]) > min_error]
        return self.visualize_mols(outliers["mol_idx"].unique().tolist(), **kwargs)

    def visualize_patologic(self, min_error: float = 5.0, **kwargs):
        grouped = self.results.groupby(by="mol_idx")["error"].agg(
            lambda x: np.min(np.abs(x))
        )
        return self.visualize_mols(
            grouped[grouped > min_error].index.to_list(), **kwargs
        )

    def visualize_good(self, max_error: float = 1.0, **kwargs):
        grouped = self.results.groupby(by="mol_idx")["error"].agg(
            lambda x: np.max(np.abs(x))
        )
        return self.visualize_mols(
            grouped[grouped < max_error].index.to_list(), **kwargs
        )
