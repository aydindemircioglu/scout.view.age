import os
import numpy as np
import cv2
import torch
import pandas as pd

import sys
sys.path.append("../")
from parameters import *
from helpers import *

from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from coral_pytorch.dataset import levels_from_labelbatch


class ScoutViewDataset(Dataset):
    def __init__(self, filepath_list, transform=None, seed=2019, returnAsDict = False, returnLabel = False, image_path = "../images/"):
        # fix dataset and loading
        np.random.seed(seed) # original code does not have this
        self.df = filepath_list
        self.transform = transform
        self.image_path = image_path
        self.returnAsDict = returnAsDict
        self.returnLabel = returnLabel
        self.metaData = ["PixelSpacing", "KVP", "Exposure", "CTDIvol", "XRayTubeCurrent", "Sex"]

        # prepare meta data
        # df = self.df
        # df["Sex"] = (df["Sex"] == "F").astype(np.float32)
        # for z in self.metaData:
        #     df[z] = df[z]/np.max(df[z])
        #     df[z] = df[z].astype(np.float32)
        # df = df.fillna(df.mean())
        # self.df = df
        #


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fimg = self.df.iloc[index]["Image"]
        v = self.df.iloc[index]
        meta = v[self.metaData].values.astype(np.float32)

        img = cv2.imread(os.path.join(self.image_path, fimg))
        if self.transform:
            img = self.transform(img)

        age = self.df.iloc[index]["Age"]
        if self.returnLabel == True:
            # need to convert age to label
            label = int(np.floor(age*4))
        else:
            label = float(age)

        if self.returnAsDict == True:
            sample = {'image': img, 'meta':meta, 'label': label}
            return sample
        else:
            return img, meta, label
        raise Exception ("Return type invalid.")


def fixData (df):
    metaData = ["PixelSpacing", "KVP", "Exposure", "CTDIvol", "XRayTubeCurrent", "Sex"]
    df["Sex"] = (df["Sex"] == "F").astype(np.float32)
    for z in metaData:
        df[z] = df[z]/np.max(df[z])
        df[z] = df[z].astype(np.float32)
    df = df.fillna(df.mean())
    print ("#NAs", df.isna().sum().sum())
    return df


class ScoutViewDataModule(pl.LightningDataModule):
    def __init__(self, data_path = "../data", image_path = "../images", returnAsDict = False, returnLabel = False, transforms_train = None, transforms_test = None, imgSize = 224, BATCH_SIZE = 32, NUM_WORKERS = 4):
        super().__init__()

        #self.data_path = data_path
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_WORKERS = NUM_WORKERS
        self.image_path = image_path
        self.train_df = getData("train", data_path)
        self.val_df = getData("val", data_path)
        self.test_df = getData("test", data_path)

        # stupid fix for NANs because of a very few meta missing
        self.train_df = fixData (self.train_df)
        self.val_df = fixData (self.val_df)
        self.test_df = fixData (self.test_df)

        self.imgSize = imgSize
        self.returnAsDict = returnAsDict
        self.returnLabel = returnLabel

        if transforms_train is None:
            transforms_train = getTransform ("train", imgSize)
        self.transforms_train = transforms_train

        if transforms_test is None:
            transforms_test = getTransform ("test", imgSize)
        self.transforms_test = transforms_test
        pass


    def prepare_data(self):
        pass


    def setup(self, stage=None):
        if stage is None:
            self.train_dataset = ScoutViewDataset(self.train_df, image_path = self.image_path, transform = self.transforms_train, returnAsDict = self.returnAsDict, returnLabel = self.returnLabel)
            self.valid_dataset = ScoutViewDataset(self.val_df, image_path = self.image_path, transform = self.transforms_test, returnAsDict = self.returnAsDict, returnLabel = self.returnLabel)
            self.test_dataset = ScoutViewDataset(self.val_df, image_path = self.image_path, transform = self.transforms_test, returnAsDict = self.returnAsDict, returnLabel = self.returnLabel)
        elif stage == "final":
            train = pd.concat([self.train_df, self.val_df])
            self.train_df = train.copy()
            self.val_df = train.copy()

            self.train_dataset = ScoutViewDataset(self.train_df, image_path = self.image_path, transform = self.transforms_train, returnAsDict = self.returnAsDict, returnLabel = self.returnLabel)
            self.valid_dataset = ScoutViewDataset(self.val_df, image_path = self.image_path, transform = self.transforms_test, returnAsDict = self.returnAsDict, returnLabel = self.returnLabel) # dummy
            self.test_dataset = ScoutViewDataset(self.test_df, image_path = self.image_path, transform = self.transforms_test, returnAsDict = self.returnAsDict, returnLabel = self.returnLabel)

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=self.BATCH_SIZE,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=self.NUM_WORKERS)
        return train_loader

    def val_dataloader(self, batch_size = None):
        if batch_size is None:
            batch_size = self.BATCH_SIZE
        valid_loader = DataLoader(dataset=self.valid_dataset,
                                  batch_size=batch_size,
                                  drop_last=False,
                                  shuffle=False,
                                  num_workers=self.NUM_WORKERS)
        return valid_loader

    def test_dataloader(self, batch_size = None):
        if batch_size is None:
            batch_size = self.BATCH_SIZE
        test_loader = DataLoader(dataset=self.test_dataset,
                                 batch_size=batch_size,
                                 drop_last=False,
                                 shuffle=False,
                                 num_workers=self.NUM_WORKERS)
        return test_loader



if __name__ == "__main__":
    transforms_train = torchvision.transforms.Compose([
        tf.ToPILImage(),
        tf.Resize((512,512)),
        tf.ToTensor(),
    ])


    dm = ScoutViewDataModule(data_path = "./data", image_path = "./images", returnLabel = False, transforms_train = transforms_train)
    dm.setup()
    train_loader = dm.train_dataloader ()
    val_loader = dm.val_dataloader ()
    test_loader = dm.test_dataloader ()

    # Checking the dataset
    def getBaseline (data_loader, split):
        all_train_labels = []
        mean, std = (0, 0)
        for images, labels in data_loader:
            all_train_labels.append(labels)
            batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
        mean /= len(data_loader.dataset)
        std /= len(data_loader.dataset)
        print ("mean, std:", mean, std)
        all_train_labels = torch.cat(all_train_labels)

        avg_prediction = torch.median(all_train_labels)  # median minimizes MAE
        baseline_mae = torch.mean(torch.abs(all_train_labels - avg_prediction))
        baseline_sd = torch.std(torch.abs(all_train_labels - avg_prediction))
        print(f'Baseline MAE@{split}: {baseline_mae:.2f} +/- {baseline_sd:.2f}')

    getBaseline (train_loader, "train")
    getBaseline (val_loader, "val")
    getBaseline (test_loader, "test")
    pass


#
