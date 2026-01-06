import lightning as L

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from uuid import uuid4


class BaseNN(L.LightningModule):
    def __init__(self, lr, pos_weight, run_id, threshold=0.5):
        super().__init__()
        # NOTE: layer definitions go in concrete classes
        self.pos_weight = pos_weight
        self.lr = lr
        self.threshold = threshold

        # Set up loss function with class weighting for imbalanced data
        if pos_weight is not None:
            self.loss_fn = BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
            )
        else:
            self.loss_fn = BCEWithLogitsLoss()

        # Use custom run_id if provided, otherwise generate UUID
        self.run_id = run_id if run_id is not None else str(uuid4())

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch):
        # Unpack batch, run forward pass, compute loss, return loss
        batch_x, batch_y = batch
        predictions = self.forward(batch_x)
        loss = self.loss_fn(predictions, batch_y)
        return loss

    def validation_step(self, batch):
        # Unpack the batch into features and labels
        batch_x, batch_y = batch

        # Forward pass - get model predictions (logits)
        predictions = self(batch_x)

        # Calculate loss for this batch
        loss = self.loss_fn(predictions, batch_y)

        # Convert logits to probabilities using sigmoid activation
        # BCEWithLogitsLoss expects logits, but metrics need probabilities
        probs = torch.sigmoid(predictions)

        # Convert probabilities to binary predictions using the object's threshold
        binary_preds = (probs > self.threshold).float()

        # Log the validation loss to Lightning's progress bar
        self.log("val_loss", loss, prog_bar=True)

        # Collect outputs for epoch-level metric calculation
        # Store predictions and targets from this batch
        output = {"preds": binary_preds, "probs": probs, "targets": batch_y}
        self.validation_step_outputs.append(output)

        return loss

    def test_step(self, batch):
        # Unpack the batch into features and labels
        batch_x, batch_y = batch

        # Forward pass - get model predictions (logits)
        predictions = self(batch_x)

        # Calculate loss for this batch
        loss = self.loss_fn(predictions, batch_y)

        # Convert logits to probabilities using sigmoid activation
        probs = torch.sigmoid(predictions)

        # Convert probabilities to binary predictions using the object's threshold
        binary_preds = (probs > self.threshold).float()

        # Log the test loss to Lightning's progress bar
        self.log("test_loss", loss, prog_bar=True)

        # Collect outputs for epoch-level metric calculation
        output = {"preds": binary_preds, "probs": probs, "targets": batch_y}
        self.test_step_outputs.append(output)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
