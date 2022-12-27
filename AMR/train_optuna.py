#!/usr/bin/python3
from pprint import pprint
import optuna
from optuna.samplers import TPESampler
import copy
import sys

import pandas as pd
import os
import time
import json
import argparse
from joblib import load, dump

import torch
from torchinfo import summary
import torchvision
from torchvision import transforms as tf
import torchvision.models as models
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import seed_everything

import random
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.nn import Conv2d, ReLU, MaxPool2d, BatchNorm2d
from torchinfo import summary
from torch import optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.models.resnet import resnet18, resnet34, resnet50
from torchvision.models.vgg import vgg16
from residue_loss_adaptive_K import MeanResidueLossAdaptive  # for K value searching , replace this with mean residue loss
import cv2
from argparse import Namespace

import sys
sys.path.append("..")
from helpers import *
from parameters import *
from model import *
from dataset import *


## stupid workaround! cannot assign variables
def objective(trial,
                K = None,
                LAMBDA_2 = None,
                freeze = None,
                gamma = None,
                imgSize = None,
                lr = None,
                modelname = None,
                nd1 = None,
                nd2 = None,
                nd3 = None,
                step_size = None,
                final = None,
                n_epochs = None):

    # we want that the trial is not dependend on the others, so seed is fixed here.
    seed_everything(1977)
    result_directory = os.path.join(rootDir, "AMR")

    # retrain or tune?
    if trial is None:
        LAMBDA_1 = 0.2 # no tuning
        LAMBDA_2 = LAMBDA_2/100
        gamma = gamma/10
        headSize = [2**nd1, 2**nd2, 2**nd3]
    else:
        n_epochs = 100
        K = trial.suggest_int('K', 3, 10)
        LAMBDA_1 = 0.2 # no tuning
        LAMBDA_2 = trial.suggest_int('LAMBDA_2', 0, 20)
        LAMBDA_2 = LAMBDA_2/100

        modelname = trial.suggest_categorical('modelname', ["densenet121", "resnet18", "resnet34", "resnet50"])
        freeze = trial.suggest_int('freeze', -1, 4)

        step_size = trial.suggest_int('step_size', 15, 30)
        gamma =  trial.suggest_int('gamma', 5, 10)
        gamma = gamma/10

        # number
        nd1 = trial.suggest_int('nd1', 2, 10)
        nd2 = trial.suggest_int('nd2', 2, 10)
        nd3 = trial.suggest_int('nd3', 2, 10)
        headSize = [2**nd1, 2**nd2, 2**nd3]

        lr = trial.suggest_float('lr', 1e-6, 0.1, log = True)
        imgSize = trial.suggest_categorical("imgSize", [224, 512])
        print (trial.params)
        recreatePath (result_directory)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_checkpoint = ModelCheckpoint(save_top_k=1, mode='min', monitor="valid_mae")
    early_stop_callback = EarlyStopping(monitor="valid_mae", min_delta=0.05, patience=20, verbose=False, mode="min")
    progress_bar = TQDMProgressBar(refresh_rate = 10)

    dm = ScoutViewDataModule(returnLabel = True, imgSize = imgSize, returnAsDict = False)
    if trial is None:
        # retrain same model
        if final == False:
            callbacks = [model_checkpoint, early_stop_callback, progress_bar]
            default_root_dir = os.path.join("../checkpoints", "AMR", "best")
            logger = TensorBoardLogger(save_dir=os.path.join("../checkpoints", "AMR", "best"), name="logs")
            dm.setup()
        else:
            callbacks = [model_checkpoint, early_stop_callback, progress_bar]
            default_root_dir = os.path.join("../checkpoints", "AMR", "final")
            logger = TensorBoardLogger(save_dir=os.path.join("../checkpoints", "AMR", "final"), name="logs")
            dm.setup("final")
    else:
        callbacks = [model_checkpoint, early_stop_callback, progress_bar]
        default_root_dir = os.path.join(rootDir, "AMR/")
        logger = TensorBoardLogger(save_dir=os.path.join(rootDir, "AMR/"), name="logs")
        dm.setup()

    lightning_model = TopoAge (learning_rate = lr, model = modelname, headSize = headSize,
            freeze = freeze, num_classes = NUM_CLASSES + 1, flatten = False, loss = "AMR",
            LAMBDA_1 = LAMBDA_1, LAMBDA_2 = LAMBDA_2, K = K,
            step_size = step_size, gamma = gamma)


    trainer = pl.Trainer(
        max_epochs = n_epochs,
        callbacks = callbacks,
        accelerator="auto",  # Uses GPUs or TPUs if available
        devices=[0],#"auto",  # Uses all available GPUs/TPUs if applicable
        logger=logger,
        default_root_dir = default_root_dir,
        deterministic=True,
        log_every_n_steps=10)

    start_time = time.time()
    trainer.fit(model=lightning_model, datamodule=dm)
    error = model_checkpoint.best_model_score
    if final != True:
        best_model = lightning_model.load_from_checkpoint(model_checkpoint.best_model_path, model = modelname)
    else:
        best_model = lightning_model

    # test only the best model-- on train this will be val, else test
    z = trainer.predict(model=best_model, dataloaders=dm.test_dataloader())
    preds = torch.cat(z).cpu().numpy()
    test_df = dm.test_dataset.df.copy()
    gt = test_df["Age"].values
    assert (preds.shape == gt.shape)
    mae = np.mean(np.abs(preds-gt))
    print ("MAE:", mae, "MAE by checkpoint", error)

    # ensure we dont have old models clogging up memory
    if trial is None:
        if final == False:
            # we retrained the model, we save the preds=val preds
            test_df["preds"] = preds
            mae = np.mean(np.abs(test_df["preds"]-test_df["Age"]))
            print ("final MAE on val:", mae)
            # we only need these
            test_df = test_df[["preds", "Age"]]
            test_df.to_csv("../results/AMR.validation.csv")
            shutil.copyfile (model_checkpoint.best_model_path, "../results/AMR.best_train.model.ckpt")
            dump(early_stop_callback, "../results/AMR.earlystopping.dump")
        else:
            test_df["preds"] = preds
            mae = np.mean(np.abs(test_df["preds"]-test_df["Age"]))
            print ("final MAE on test:", mae)
            test_df = test_df[["preds", "Age"]]
            test_df.to_csv("../results/AMR.test.csv")
            shutil.copyfile (model_checkpoint.best_model_path, "../results/AMR.final.model.ckpt")
        return None
    else:
        trainer.test(model=best_model, dataloaders=dm.val_dataloader())
        pass

    import gc
    best_model.cpu()
    lightning_model.cpu()
    del best_model, trainer, lightning_model
    gc.collect()
    torch.cuda.empty_cache()
    return error



if __name__ == "__main__":
    print ("### Experiment using AMR Loss")
    os.makedirs ("../results", exist_ok = True)
    sampler = TPESampler(seed=634)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(df)
    df.to_csv(f"../results/optuna.results.AMR.csv")


#
