from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningModule, LightningDataModule
from modules.datamodules import AtomicShiftsDatamodule
from modules.models import SimpleReadOut
import lightning.pytorch.callbacks
from torch import set_float32_matmul_precision


def cli_main():
    cli = LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        seed_everything_default=42,
        auto_configure_optimizers=False,
    )
    # note: don't call fit!!


if __name__ == "__main__":
    set_float32_matmul_precision("high")
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
