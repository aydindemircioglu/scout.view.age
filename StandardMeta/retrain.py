#!/usr/bin/python3
from pprint import pprint
import optuna
from optuna.samplers import TPESampler
import copy

import pandas as pd
import os
import time
import json
import argparse

import torch
from torchinfo import summary
import torchvision
from torchvision import transforms as tf
import torchvision.models as models
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
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
import cv2

import sys
sys.path.append("..")
from helpers import *
from parameters import *
from model import *
from dataset import *
from train_optuna import objective

from joblib import load, dump





if __name__ == "__main__":
    print ("### Experiment using Standard+Meta architecture")
    tResults = pd.read_csv("../results/optuna.results.StandardMeta.csv")
    bestModel = tResults[tResults.value == np.min(tResults.value)].iloc[0]
    # TOOD: add this to parameters
    best_params = {k.replace("params_", ''):bestModel[k] for k in bestModel.keys() if "params_" in k}
    model = objective(None, **best_params, n_epochs = 100, final = False)
    early_stop_callback = load("../results/StandardMeta.earlystopping.dump")
    n_epochs = early_stop_callback.stopped_epoch
    model = objective(None, **best_params, n_epochs = n_epochs, final = True)


#
