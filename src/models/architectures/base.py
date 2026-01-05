import lightning as L
import torch


class Base(L.LightningModule):
    def __init__(self):
        super().__init__()
        # TODO: binary classifier, so BCEWithLogitLoss for pos_weight?
        # TODO: learning rate and threshold
        # TODO: pos_weight (if necessary)
        # TODO: run_id
        # TODO: test and validation outputs (lists)
        # NOTE: layer definitions go in concrete classes
        pass

    def training_step(self):
        # NOTE: forward() method goes in concrete classes
        pass

    def validation_step(self):
        pass

    def test_step(self):
        pass

    def configure_optimizers(self):
        pass
