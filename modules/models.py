import torch
from torch import nn
from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer, seed_everything
from torch.nn import functional as F
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from torch_geometric.nn.models import GCN, GraphSAGE, GAT


class FFN(nn.Module):
    def __init__(
        self,
        layers: list,
        activation: str,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = []
        input = layers[0]
        output = layers[-1]
        activations = nn.ModuleDict(
            [
                ["leakyrelu", nn.LeakyReLU()],
                ["relu", nn.ReLU()],
                ["elu", nn.ELU()],
                ["selu", nn.SELU()],
            ]
        )
        self.activation = activations[activation]
        for layer in layers[1:-1]:
            self.layers.extend([nn.Linear(input, layer), self.activation])
            self.layers.append(nn.Dropout(dropout))
            input = layer
        self.layers.append(nn.Linear(input, output))
        self.dense = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.dense(x)


class SimpleReadOut(LightningModule):
    """
    layers: list of layer widths (including input and output layers)
    activation: relu, leakyrelu or elu (default elu)
    lr: learning rate (default 1e-3)
    dropout: dropout rate in all layers
    loss: type of loss: mae, mse or huber (default mae)
    delta: huber loss parameter (default 1.0) *only needed if the loss is huber*
    """

    def __init__(
        self,
        layers: list,
        activation: str = "elu",
        dropout: float = 0.0,
        loss: str = "mae",
        delta: float = 1.0,
        optimizer: OptimizerCallable = lambda p: torch.optim.AdamW(
            p, lr=3e-4, weight_decay=0.01
        ),
        lr_scheduler: LRSchedulerCallable = lambda p: torch.optim.lr_scheduler.ExponentialLR(
            p, gamma=1.0
        ),
    ):
        super().__init__()
        self.model = FFN(
            layers,
            activation,
            dropout=dropout,
        )
        losses = nn.ModuleDict(
            [
                ["mae", nn.L1Loss()],
                ["mse", nn.MSELoss()],
                ["huber", nn.HuberLoss(delta=delta)],
            ]
        )
        self.loss = losses[loss]
        self.save_hyperparameters(ignore=["optimizer", "lr_scheduler"])
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def training_step(self, batch):
        x, y = batch["encoding"], batch["shift"]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch["encoding"], batch["shift"]
        y_hat = self.model(x)
        valid_loss = F.l1_loss(y_hat, y)
        self.log("val_loss", valid_loss, prog_bar=False, on_epoch=True, sync_dist=True)

    def on_test_start(self) -> None:
        self.test_results = pd.DataFrame()

    def test_step(self, batch):
        mol_idxs, atom_idxs, x, y = (
            batch["mol_idx"],
            batch["atom_idx"],
            batch["encoding"],
            batch["shift"],
        )
        y_hat = self.model(x)
        test_loss = F.l1_loss(y_hat, y)
        self.log("test_loss", test_loss, prog_bar=False, on_epoch=True)
        self.test_results = pd.concat(
            [
                pd.DataFrame(
                    {
                        "mol_idx": torch.squeeze(mol_idxs).cpu().numpy(),
                        "atom_idx": torch.squeeze(atom_idxs).cpu().numpy(),
                        "predicted_shift": torch.squeeze(y_hat).detach().cpu().numpy(),
                        "true_shift": torch.squeeze(y).cpu().numpy(),
                        "error": torch.squeeze(y_hat - y).detach().cpu().numpy(),
                    }
                ),
                self.test_results,
            ],
            ignore_index=True,
        )

    def on_test_end(self):
        self.test_results.to_csv("test.csv")

    def on_predict_start(self) -> None:
        self.results = pd.DataFrame()

    def predict_step(self, batch):
        mol_idxs, atom_idxs, x = (
            batch["mol_idx"],
            batch["atom_idx"],
            batch["encoding"],
        )
        y_hat = self.model(x)
        data = {k: batch[k].detach().cpu().numpy() for k in batch}
        data["encoding"] = [i for i in data["encoding"]]
        data.pop("shift")
        distances = self.distance_examples_from_train_set(
            pd.DataFrame(data).astype(
                {"atom_idx": np.int32, "mol_idx": np.int32, "encoding": object}
            ),
            self.trainer.datamodule.df_train.loc[
                :, ["mol_idx", "atom_idx", "encoding"]
            ],
        )
        self.results = pd.concat(
            [
                pd.merge(
                    pd.DataFrame(
                        {
                            "mol_idx": torch.squeeze(mol_idxs).cpu().numpy(),
                            "atom_idx": torch.squeeze(atom_idxs).cpu().numpy(),
                            "predicted_shift": torch.squeeze(y_hat)
                            .detach()
                            .cpu()
                            .numpy(),
                        }
                    ),
                    distances,
                    on=["mol_idx", "atom_idx"],
                    how="left",
                ),
                self.results,
            ],
            ignore_index=True,
        )

    def on_predict_end(self) -> None:
        self.results.to_csv("predict.csv")

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler = self.lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class SimpleGNN(LightningModule):
    """
    gnn_type: type of message passing layers: gcn, graphSAGE or gat
    in_channels: dimension of input node features; should be equal to the pre-trained embedding dimension
    out_channels: dimension of output node features
    hidden_chennels: dimension of node representation in hidden layers
    num_hidden_gnn: number of message passing layers
    layers_readout: list of layer widths (including input and output layers) for readout network
    activation: relu, leakyrelu or elu (default elu)
    lr: learning rate (default 1e-3)
    dropout: dropout rate in all layers
    loss: type of loss: mae, mse or huber (default mae)
    delta: huber loss parameter (default 1.0) *only needed if the loss is huber*
    """

    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_hidden_gnn: int,
        layers_readout: list,
        activation: str,
        dropout: float = 0.0,
        delta: float = 1.0,
        loss: str = "mae",
        optimizer: OptimizerCallable = lambda p: torch.optim.AdamW(
            p, lr=3e-4, weight_decay=0.01
        ),
        lr_scheduler: LRSchedulerCallable = lambda p: torch.optim.lr_scheduler.ExponentialLR(
            p, gamma=1.0
        ),
    ):
        super().__init__()
        gnns = {
            "gcn": GCN,
            "graphSAGE": GraphSAGE,
            "gat": GAT,
        }
        self.GNN = gnns[gnn_type](
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_hidden_gnn,
            act=activation,
            dropout=dropout,
        )

        self.readout = FFN(
            layers_readout,
            activation,
            dropout=dropout,
        )
        losses = nn.ModuleDict(
            [
                ["mae", nn.L1Loss()],
                ["mse", nn.MSELoss()],
                ["huber", nn.HuberLoss(delta=delta)],
            ]
        )
        self.loss = losses[loss]
        self.save_hyperparameters(ignore=["optimizer", "lr_scheduler"])
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, x, edge_index, edge_attr):
        return self.readout(self.GNN(x, edge_index))

    def training_step(self, batch):
        x, edge_index, edge_attr, y, mask = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.y,
            batch.mask,
        )
        y_hat = self.forward(x, edge_index, edge_attr)
        loss = self.loss(y_hat[mask], y[mask])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, edge_index, edge_attr, y, mask = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.y,
            batch.mask,
        )
        y_hat = self.forward(x, edge_index, edge_attr)
        valid_loss = F.l1_loss(y_hat[mask], y[mask])
        self.log("val_loss", valid_loss, prog_bar=False, on_epoch=True, sync_dist=True)

    def on_test_start(self) -> None:
        self.test_results = pd.DataFrame()

    def test_step(self, batch):
        x, edge_index, edge_attr, y, mask = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.y,
            batch.mask,
        )
        y_hat = self.forward(x, edge_index, edge_attr)
        test_loss = F.l1_loss(y_hat[mask], y[mask])
        self.log("test_loss", test_loss, prog_bar=False, on_epoch=True)
        self.test_results = pd.concat(
            [
                pd.DataFrame(
                    {
                        "mol_idx": torch.squeeze(batch.mol_id[mask]).cpu().numpy(),
                        "atom_idx": torch.squeeze(batch.atom_id[mask]).cpu().numpy(),
                        "predicted_shift": torch.squeeze(y_hat[mask])
                        .detach()
                        .cpu()
                        .numpy(),
                        "true_shift": torch.squeeze(y[mask]).cpu().numpy(),
                        "error": torch.squeeze(y_hat[mask] - y[mask]).cpu().numpy(),
                    }
                ),
                self.test_results,
            ],
            ignore_index=True,
        )

    def on_test_end(self):
        self.test_results.to_csv("test.csv")

    def on_predict_start(self) -> None:
        self.results = pd.DataFrame()

    def predict_step(self, batch):

        x, edge_index, edge_attr, mask = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.mask,
        )

        y_hat = self.forward(x, edge_index, edge_attr)
        self.results = pd.concat(
            [
                pd.DataFrame(
                    {
                        "mol_idx": torch.squeeze(batch.mol_id[mask]).cpu().numpy(),
                        "atom_idx": torch.squeeze(batch.atom_id[mask]).cpu().numpy(),
                        "predicted_shift": torch.squeeze(y_hat[mask])
                        .detach()
                        .cpu()
                        .numpy(),
                    }
                ),
                self.results,
            ],
            ignore_index=True,
        )

    def on_predict_end(self) -> None:
        self.results.to_csv("predict.csv")
        return self.results

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler = self.lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
