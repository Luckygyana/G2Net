import pytorch_lightning as pl
import torch.nn as nn
import torch
import timm

from sklearn.metrics import roc_auc_score

from src.utils.technical_utils import load_obj


class G2Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(
            self.cfg.model.model_name,
            pretrained=self.cfg.model.pretrained,
            in_chans=self.cfg.model.inp_channels,
        )

        if "resent" in self.cfg.model.model_name:
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, self.cfg.model.out_features, bias=True)

        if "nfnet" in self.cfg.model.model_name:
            n_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(n_features, self.cfg.model.out_features, bias=True)

        elif "eff" in self.cfg.model.model_name:
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, self.cfg.model.out_features, bias=True)

        self.criterion = load_obj(cfg.loss.class_name)()
        self.batch_size = self.cfg.datamodule.train_batch_size
    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):

        x, labels = batch
        output = self.model(x)
        loss = self.criterion(output.view(-1), labels)

        try:
            auc = roc_auc_score(labels.detach().cpu(), output.sigmoid().detach().cpu())
            self.log("auc", auc, on_step=True, prog_bar=True, logger=True)
            self.log("Train Loss", loss, on_step=True, prog_bar=True, logger=True)
        except:
            pass

        return {"loss": loss, "preds": output, "targets": labels}

    def training_epoch_end(self, outputs):

        preds = []
        labels = []

        for output in outputs:

            preds += output["preds"]
            labels += output["targets"]

        labels = torch.stack(labels)
        preds = torch.stack(preds)

        train_auc = roc_auc_score(labels.detach().cpu(), preds.sigmoid().detach().cpu())
        self.log("train_auc", train_auc, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        output = self.model(x)
        loss = self.criterion(output.view(-1), labels)

        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {"preds": output, "targets": labels}

    def validation_epoch_end(self, outputs):

        preds = []
        labels = []

        for output in outputs:
            preds += output["preds"]
            labels += output["targets"]

        labels = torch.stack(labels)
        preds = torch.stack(preds)

        val_auc = roc_auc_score(labels.detach().cpu(), preds.sigmoid().detach().cpu())
        self.log("val_auc", val_auc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = load_obj(self.cfg.optimizer.class_name)(
            self.model.parameters(), **self.cfg.optimizer.params
        )
        scheduler = load_obj(self.cfg.scheduler.class_name)(optimizer, **self.cfg.scheduler.params)

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": self.cfg.scheduler.step,
                    "monitor": self.cfg.scheduler.monitor,
                }
            ],
        )
