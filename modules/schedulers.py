from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR


class CustomLR(MultiplicativeLR):
    def __init__(
        self,
        optimizer,
        decay_rate: float = 0.96,
        period: int = 15,
        stop_epoch: int = 1000,
    ):
        lr_lambda = lambda x: (
            decay_rate if (x % period == period - 1) and x <= stop_epoch else 1.0
        )
        super().__init__(optimizer, lr_lambda)
