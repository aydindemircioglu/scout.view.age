#
import argparse
from pprint import pprint
import numpy as np
from glob import glob
import pandas as pd
import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image

from dataset import *
from model import *
from evaluate import *

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Saliency
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from torch import nn
from PIL import ImageDraw, ImageFont
import PIL
from captum.attr import Occlusion


def mycropImage(img):
    #img = images[5]["rtC"]
    k = img[:,:,0]
    try:
        cut0 = np.where(np.std(k, axis = 1) < 10)[0][0]
    except:
        return img

    # is the top empty?
    if cut0 < 32:
        try:
            cut0 = np.where(np.std(k, axis = 1) < 10)[0]
            lidx = np.where((cut0[1:] - cut0[:-1]) > 2)[0][0]
            kimg = k.copy()*0
            kimg[0:k.shape[0]-cut0[lidx]] = k[cut0[lidx]:,:]
            k = kimg.copy()
            nimg = img.copy()*0
            nimg[0:k.shape[0]-cut0[lidx],:] = img[cut0[lidx]:,:,:]
            img = nimg.copy()
        except Exception as e:
            print(e)
            return img
    try:
        cut1 = np.where(np.std(k, axis = 1) < 10)[0][0]
    except Exception as e:
        return img

    cuth2 = (k.shape[0] - cut1)//2
    if cuth2 > 128:
        cut1 = k.shape[0] - 222
        cuth2 = 111

    k = img[0:cut1, cuth2:(k.shape[1]-cuth2), :]
    k = cv2.resize(k, (512, 512))
    return (k)


if __name__ == '__main__':
    # get best params of best model = AMR
    tResults = pd.read_csv("./results/optuna.results.AMR.csv")
    bestModel = tResults[tResults.value == np.min(tResults.value)].iloc[0]
    best_params = {k.replace("params_", ''):bestModel[k] for k in bestModel.keys() if "params_" in k}
    for k in best_params:
        exec(k + "=" + repr(best_params[k]))
    LAMBDA_1 = 0.2 # no tuning
    LAMBDA_2 = LAMBDA_2/100
    gamma = gamma/10
    headSize = [2**nd1, 2**nd2, 2**nd3]

    dm = ScoutViewDataModule(returnLabel = True, data_path = "./data", image_path = "./images", imgSize = imgSize, returnAsDict = False)
    dm.setup("final")

    lightning_model = TopoAge (learning_rate = lr, model = modelname, headSize = headSize,
            freeze = freeze, num_classes = NUM_CLASSES + 1, flatten = False, loss = "AMR",
            LAMBDA_1 = LAMBDA_1, LAMBDA_2 = LAMBDA_2, K = K,
            step_size = step_size, gamma = gamma)

    model_path = os.path.join("./checkpoints", "AMR", "final", "logs", "version_0", "checkpoints", "*")
    model_path = glob(model_path)[0]
    model = lightning_model.load_from_checkpoint(model_path, learning_rate = lr, model = modelname, headSize = headSize,
            freeze = freeze, num_classes = NUM_CLASSES + 1, flatten = False, loss = "AMR",
            LAMBDA_1 = LAMBDA_1, LAMBDA_2 = LAMBDA_2, K = K,
            step_size = step_size, gamma = gamma)

    _ = model.cuda()
    _ = model.eval()

    # strange place, but trained already. print  best model parameters size
    print(summary(model, input_size=(1, 3, imgSize, imgSize), verbose = 0))


    random.seed(1977)
    df = dm.test_df.copy()
    df = df.reset_index()

    images = []
    ageRange = list(np.arange(0.5,21,1.8))
    for a in ageRange:
        subdf = df.query("age >= @a")
        subdf = subdf.sort_values(["age"])
        images.append({"key": subdf.iloc[0]["index"], "age": subdf.iloc[0]["age"]})


    for j, batch in enumerate(dm.test_dataloader(batch_size = 1)):
        ij = -1
        for l, k in enumerate(images):
            if k["key"] == j:
                ij = l
        if ij == -1:
            continue

        y = batch[1].cuda()
        torch_img = batch[0].cuda()

        # its AMR
        occlusion = Occlusion(model)
        attributions_occ = occlusion.attribute(torch_img.cuda(),
                                               strides = (3, 8, 8),
                                               target=y,
                                               sliding_window_shapes=(3, 48, 48),
                                               baselines=0)

        transformed_img = (torch_img - torch.min(torch_img))/(torch.max(torch_img) - torch.min(torch_img))
        rt = np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0))
        s = (rt[:,:,0] + rt[:,:,1] + rt[:,:,2])/3.0
        rt[:,:,0] = s; rt[:,:,1] = s; rt[:,:,2] = s
        rt = rt/np.max(rt)*255
        rtA = np.asarray(rt, dtype = np.uint8)

        rt = np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0))
        rtmax = np.max([np.abs(np.min(rt)), np.abs(np.max(rt))])
        rt = (rt + rtmax)/(2*rtmax)
        rt = rt*255

        rtB = np.asarray(rt, dtype = np.uint8)
        rtB = cv2.applyColorMap(rtB, cv2.COLORMAP_TWILIGHT_SHIFTED)

        rtC = 1.0*rtA + rtB*0.5
        rtC = rtC/(np.max(rtC))
        rtC = np.asarray(255*rtC, dtype = np.uint8)
        rImg = np.hstack([rtA,rtB,rtC])
        images[ij]["pred"] = y.cpu().numpy()
        images[ij]["rtA"] = rtA
        images[ij]["rtB"] = rtB
        images[ij]["rtC"] = rtC
        images[ij]["rImg"] = rImg

    sp = 16
    fx = 4; fy = 3
    finalImg = np.zeros((512*fy+sp*fy+sp, 512*fx+sp*fx+sp, 3), dtype = np.uint8)
    finalImg = 255 + finalImg

    MAE, testResults = getPreds ("AMR", "test")
    for j, k in enumerate(images):
        o1 = sp + (sp+512)*(j//fx)
        o2 = sp + (sp+512)*(j%fx)
        k = mycropImage(images[j]["rtC"])
        #k = images[j]["rtC"]
        age = np.round(testResults.iloc[images[j]["key"]]["Age"],1)
        pred = np.round(testResults.iloc[images[j]["key"]]["test_preds"],1)
        k[460:,:,:] = 0
        print (age, pred)
        k = addText (k, text="True: "+str(age) + "y, Prediction: " + str(pred) +"y", org = (20, 465), fontSize = 36)
        finalImg[o1:o1+512,o2:o2+512,:] = k


    z = cv2.resize(finalImg, (0,0), fx=2.5, fy = 2.5)
    cv2.imwrite("./paper/Figure_8.png", z)

#
