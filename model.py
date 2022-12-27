#
import argparse
from pprint import pprint
import numpy as np
import pandas as pd
import os
import random

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as tf
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torchvision.models as models
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchinfo import summary
from pytorch_lightning import seed_everything
import torchmetrics

from coral_pytorch.losses import corn_loss, coral_loss
from coral_pytorch.dataset import corn_label_from_logits, levels_from_labelbatch, proba_to_label
from coral_pytorch.layers import CoralLayer

#from datasets import *
from parameters import *
from AMR.residue_loss_adaptive_K import MeanResidueLossAdaptive  # for K value searching , replace this with mean residue loss


class TopoAge(pl.LightningModule):
    def __init__(self, learning_rate = 3e-4, imgSize = 224, num_classes = 1, model = None, headSize = [512, 256, 64],
                step_size = 1, gamma = 0.9, flatten = True,
                LAMBDA_1 = None, LAMBDA_2 = None, K = None,
                dropoutLevel = 0.1, freeze = None, showStats = True, loss = "L1"):
        super().__init__()
        self.headSize = headSize
        self.dropoutLevel = dropoutLevel
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.lossType = loss
        self.model = model
        self.flatten = flatten
        self.num_classes = num_classes
        self.LAMBDA_1 = LAMBDA_1
        self.LAMBDA_2 = LAMBDA_2
        self.K = K

        if self.lossType == "AMR":
            self.criterion1 = MeanResidueLossAdaptive(self.LAMBDA_1, self.LAMBDA_2, START_AGE, END_AGE, self.K)
            self.criterion2 = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters(ignore="model")

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        if model == "densenet121":
            num_filters = 1024
            if freeze == -1:
                backbone = models.densenet121()
                backbone.classifier = nn.Identity()
            else:
                backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
                backbone.classifier = nn.Identity()
                freezeToInt = [0, 6, 8, 10, 12]
                for name in backbone.children():
                  for j, (child, config) in enumerate(name.named_children()):
                    if j >= freezeToInt[freeze]:
                      #print(str(child) + ' is unfrozen')
                      for param in config.parameters():
                          param.requires_grad = True
                    else:
                      #print(str(child) + ' is frozen')
                      for param in config.parameters():
                          param.requires_grad = False
            self.feature_extractor = backbone

 
        if "resnet" in model:
            fmodel = "ResNet" + model[6:]
            if freeze == -1:
                backbone = eval("models."+model+"()")
                num_filters = backbone.fc.in_features
                backbone.fc = nn.Identity()
            else:
                backbone = eval("models."+model+"(weights=models."+fmodel+"_Weights.IMAGENET1K_V1)")
                num_filters = backbone.fc.in_features
                backbone.fc = nn.Identity()

                #freezeToInt = [0, 39, 81, 143, 159][freeze]
                if model == "resnet18":
                    freezeToInt = [-1, 14, 29, 44, 64][freeze]
                if model == "resnet34":
                    freezeToInt = [-1, 20, 47, 86, 133][freeze]
                if model == "resnet50":
                    freezeToInt = [-1, 32, 71, 128, 159][freeze]
                for param in list(backbone.parameters())[:freezeToInt+1]:
                    param.requires_grad=False
                # backbone_layers = list(backbone.children())[:-1]
                # frozen_layers = []
                # trainable_layers = backbone_layers[freezeToInt:]
                # for layer in backbone_layers[:freezeToInt]:
                #     for param in layer.parameters():
                #         param.requires_grad = False
                #     frozen_layers.append(layer)
                # flayers = frozen_layers + trainable_layers
                # backbone = nn.Sequential(*flayers)
            self.feature_extractor = backbone


        # create head
        clf = []
        clf.append(nn.Dropout(self.dropoutLevel))
        cSize = num_filters
        for j, z in enumerate(headSize):
            clf.append(nn.Linear(cSize, z))
            clf.append(nn.ReLU())
            cSize = z
        if self.lossType == "Coral":
            clf.append(CoralLayer(size_in=cSize, num_classes=self.num_classes))
        else:
            clf.append(nn.Linear(cSize, self.num_classes))
        self.classifier = nn.Sequential(*clf)


    def forward(self, x):
        fv = self.feature_extractor(x)
        fv = fv.flatten(1)
        x = self.classifier(fv)
        if self.flatten == True:
            x = x.flatten()
        return x


    def _shared_step(self, batch):
        x, y = batch
        y_hat = self(x)

        if self.lossType == "AMR":
            mean_loss, variance_loss, adaptive_K = self.criterion1(y_hat, y)
            softmax_loss = self.criterion2(y_hat, y)
            loss = mean_loss + variance_loss + softmax_loss

            m = nn.Softmax(dim=1)
            y_hat = m(y_hat)
            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).to(y.device)
            mean = (y_hat * a).sum(1, keepdim=True)
            y_hat = torch.round(mean) # label --> 4x larger
            y_hat = y_hat.flatten()

        if self.lossType == "L1":
            loss = F.l1_loss(y_hat, y)

        if self.lossType == "Coral":
            # levels land on cpu....
            levels = levels_from_labelbatch(y, num_classes = self.num_classes)
            levels = levels.to(y.device)
            loss = coral_loss (y_hat, levels)
            probas = torch.sigmoid(y_hat)
            y_hat = proba_to_label(probas)

        if self.lossType == "Corn":
            # network is C-1, but loss wants to know the correct #classes
            #print (y_hat.size(), y.size(), self.num_classes)
            loss = corn_loss(y_hat, y, num_classes=self.num_classes+1)
            y_hat = corn_label_from_logits(y_hat)

        return loss, y, y_hat


    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self._shared_step (batch)
        self.log("train_loss", loss)
        if self.lossType == "AMR" or self.lossType == "Corn" or self.lossType == "Coral":
            self.train_mae(y_hat/4+1/8, y/4) # true MAE now
        elif self.lossType == "L1":
            self.train_mae(y_hat, y)
        else:
            raise Exception ("Unknown loss")
        self.log("train_mae", self.train_mae, on_epoch=True, on_step=False)
        return loss  # this is passed to the optimzer for training


    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self._shared_step (batch)
        self.log("valid_loss", loss)
        if self.lossType == "AMR" or self.lossType == "Corn" or self.lossType == "Coral":
            self.valid_mae(y_hat/4+1/8, y/4) # true MAE now
        elif self.lossType == "L1":
            self.valid_mae(y_hat, y)
        else:
            raise Exception ("Unknown loss")
        self.log("valid_mae", self.valid_mae, on_epoch=True, on_step=False, prog_bar=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch

        if self.lossType == "Coral":
            # levels land on cpu....
            logits = self(x)
            probas = torch.sigmoid(logits)
            y_hat = proba_to_label(probas)
            return y_hat/4+1/8

        if self.lossType == "Corn":
            logits = self(x)
            y_hat = corn_label_from_logits(logits)
            return y_hat/4+1/8

        if self.lossType == "AMR":
            y_hat = self(x)
            m = nn.Softmax(dim=1)
            y_hat = m(y_hat)
            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).to(x.device)
            mean = (y_hat * a).sum(1, keepdim=True)
            y_hat = torch.round(mean) # label --> 4x larger
            y_hat = y_hat.flatten()
            return y_hat/4+1/8

        if self.lossType == "L1":
            y_hat = self(x)
            return y_hat

        raise Exception ("Unknown loss")


    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self._shared_step (batch)
        if self.lossType == "AMR" or self.lossType == "Corn" or self.lossType == "Coral":
            self.test_mae(y_hat/4+1/8, y/4) # true MAE now
        else:
            self.test_mae(y_hat, y)
        self.log("test_mae", self.test_mae, on_epoch=True, on_step=False)


    def configure_optimizers(self):
        print ("Setting learning rate to", self.learning_rate)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = self.step_size, gamma = self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}


if __name__ == "__main__":
    # just browsing
    pTbl = {}
    for model in ["densenet121", "resnet18", "resnet34", "resnet50"]:
        print ("\n\n###", model)
        pTbl[model] = {}
        for imgSize in [224, 512]:
            for freeze in [-1, 0, 1, 2, 3, 4]:
                m = TopoAge(model = model, freeze = freeze, showStats = True, headSize = [1024])
                print(summary(m.feature_extractor, input_size=(1, 3, imgSize, imgSize), verbose = 0))
                z = summary(m.feature_extractor, input_size=(1, 3, imgSize, imgSize), verbose = 1)
                pTbl[model][(imgSize,freeze, "Trainable")] =  '{:,.0f}'.format(z.trainable_params)
                pTbl[model][(imgSize,freeze, "Non-trainable")] = '{:,.0f}'.format(z.total_params- z.trainable_params)
    pd.DataFrame(pTbl).to_excel("./results/paramCount.xlsx")



#
