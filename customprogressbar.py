import pytorch_lightning as pl
from tqdm.auto import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar

class CustomProgressBar(pl.callbacks.ProgressBar):

    def __init__(self) -> None:
        """
        Constructor for the custom progress bar used to display model metrics.
        """
        super().__init__()
        self.bar = None
        self.enabled = True

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """
        Initialize a new progress bar at the start of each training epoch.

        Args:
            trainer (Trainer): the PyTorch Lightning trainer instance
            pl_module (LightningModule): the LightningModule being trained
        """
        if self.enabled:
            self.bar = tqdm(total=self.total_train_batches,
                            desc=f"Epoch {trainer.current_epoch+1}",
                            position=0,
                            leave=True)
            self.running_loss = 0.0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """
        Update progress bar after each training batch with current metrics.

        Args:
            trainer (Trainer): the PyTorch Lightning trainer instance
            pl_module (LightningModule): the LightningModule being trained
            outputs (dict): dictionary containing batch outputs
            batch (torch.Tensor): the batch of data used in training
            batch_idx (int): index of the current batch
        """
        if self.bar:
            self.running_loss += outputs['loss'].item()
            self.bar.update(1)
            loss = self.running_loss / self.total_train_batches
            self.bar.set_postfix(loss=f'{loss:.3f}')

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """
        Update progress bar after each validation batch with current metrics.

        Args:
            trainer (Trainer): the PyTorch Lightning trainer instance
            pl_module (LightningModule): the LightningModule being trained
        """
        if self.bar:
            val_loss = trainer.callback_metrics.get('val_loss')
            loss = val_loss.item()
            self.bar.set_postfix(loss=f'{loss:.3f}', val_loss=f'{val_loss:.3f}')
            self.bar.close()
            self.bar = None

    def disable(self) -> None:
        """
        Disable the progress bar.
        """
        self.bar = None
        self.enabled = False