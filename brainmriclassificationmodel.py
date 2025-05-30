import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet50, resnet101
from torchmetrics import Accuracy
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall

class BrainMRIClassificationModel(pl.LightningModule):

    def __init__(self, model='resnset50', num_features=2, learning_rate=1e-4) -> None:
        """
        Constructor for the ResNet classification model.

        Args:
            model (str): the name of the model to use
            num_features (int): the number of features the model classifies
            learning_rate (float): the learning rate of the model
        """
        super().__init__()
        self.model = resnet101() if model == 'resnet101' else resnet50()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_features)
        self.criterion = nn.CrossEntropyLoss()

        task_type = 'binary' if num_features == 2 else 'multiclass'
        self.accuracy = Accuracy(task=task_type)
        self.learning_rate = learning_rate
        self.test_metrics = MetricCollection({
            'accuracy': Accuracy(task=task_type),
            'precision': Precision(task=task_type),
            'recall': Recall(task=task_type),
        })
        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): input tensor to the model

        Returns:
            - torch.Tensor: the model's output logits
        """
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Model training step

        Args:
            batch (tuple): tuple containing the images and labels for
            model training
            batch_idx (int): index of the current batch

        Returns:
            - torch.Tensor: computed model loss
        """
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Validation step with metric calculations and logging

        Args:
            batch (tuple): tuple containing the images and labels for
            model training
            batch_idx (int): index of the current batch

        Returns:
            - torch.Tensor: computed model loss
        """
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        accuracy = self.accuracy(preds, labels)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        return { 'val_loss': loss, 'val_acc': accuracy }

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Test step with metric accumulation
        """
        images, labels = batch
        y_hat = self(images)
        loss = self.criterion(y_hat, labels)

        preds = torch.argmax(y_hat, dim=1)
        self.test_metrics.update(preds, labels)

        return loss

    def on_test_epoch_end(self) -> None:
        """
        Compute and log metric for rank 0 at the end of testing
        """
        
        # Print only from rank 0
        if not self.trainer.is_global_zero:
            return

        metrics = self.test_metrics.compute()
        self.log_dict({f'test_{k}': v for k, v in metrics.items()})
        self.test_metrics.reset()

    def configure_optimizers(self) -> tuple[list, list]:
        """
        Configure the optimizer and learning rate scheduler

        Returns:
            - Tuple[list, list]: a tuple containing a list of optimizers and schedulers
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]
        