#

from pprint import pprint
import cv2
import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import random
import math
from tbparse import SummaryReader
import scipy

from helpers import *
from PIL import Image



def cropImage (k, crop = None, ofs = (0,0)):
    kf = np.asarray(k, dtype = np.float32)
    s = np.asarray((kf[:,:,2] + kf[:,:,1] + kf[:,:,0])/3, dtype = np.float32)
    s = np.asarray( 255*(s - np.min(s))/(np.max(s) - np.min(s)), dtype = np.uint8)
    k[:,:,0] = s
    k[:,:,1] = s
    k[:,:,2] = s
    k = k[ofs[0]:ofs[0]+crop[0], ofs[1]:ofs[1]+crop[1]]
    return k


def addText (finalImage, text = '', org = (0,0), fontFace = 'Arial', fontSize = 12, color = (255,255,255)):
     # Convert the image to RGB (OpenCV uses BGR)
     from PIL import Image
     from PIL import ImageDraw, ImageFont
     tmpImg = cv2.cvtColor(finalImage, cv2.COLOR_BGR2RGB)
     pil_im = Image.fromarray(tmpImg)
     draw = ImageDraw.Draw(pil_im)
     font = ImageFont.truetype(fontFace + ".ttf", fontSize)
     draw.text(org, text, font=font, fill = color)
     tmpImg = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
     return (tmpImg.copy())


def generateFigureExampleCTSV (loc1, loc2, results):
    finalImg = np.zeros ((512+2*16, 2*512+3*16, 3), dtype = np.uint8)
    finalImg = finalImg*0 + 255
    data = results["train"].copy()
    if type(loc1) == list:
        for idx, l in enumerate(loc1):
            print (data.iloc[l].Sex, data.iloc[l].age, "./images/"+data.iloc[l].Image)
            k = cv2.imread("./images/"+data.iloc[l].Image)
            k = cropImage(k, crop = (248,512))
            d = k.shape[1] - 512
            k = k[0:248, d//2:d//2+512]
            start = 16 + idx*(248+16)
            end = 248+16 + idx*(248 + 16)
            print (start, end, end - start)
            finalImg [start:end, 16:512+16, :] = k[0:248,:,:]

    print (data.iloc[loc2].Sex, data.iloc[loc2].age, "./images/"+data.iloc[loc2].Image)
    k = cv2.imread("./images/"+data.iloc[loc2].Image)
    k = cropImage(k, crop = (248,248), ofs = (0,128))
    k = cv2.resize(k, (512, 512))
    finalImg [16:512+16, 512+32:512+32+512, :] = k

    z = finalImg.copy()
    finalImg = cv2.resize(z, (0,0), fx = 2.5, fy = 2.5)
    if type(loc1) == list:
        t = addText (finalImg, "(A)", fontFace = "Arial", org = (1239, 1250-2.5*(248+16)), fontSize = 48, color = (255, 255, 255))
        t = addText (t, "(B)", fontFace = "Arial", org = (1239,1250), fontSize = 48, color = (255, 255, 255))
        t = addText (t, "(C)", fontFace = "Arial", org = (2559,1250), fontSize = 48, color = (255, 255, 255))
    else:
        t = addText (finalImg, "(A)", fontFace = "Arial", org = (1239,1250), fontSize = 48, color = (255, 255, 255))
        t = addText (t, "(B)", fontFace = "Arial", org = (2559,1250), fontSize = 48, color = (255, 255, 255))
    return (t)


def generateTopo(results):
    # these were actually the first ones, but train is now shuffled..
    k = generateFigureExampleCTSV ([17, 277], 1183, results)
    cv2.imwrite("paper/Figure_2.png", k)



def plotAgeHistogramm(results):
    # we need this for patients..
    table1 = {}

    f, axs  = plt.subplots(1, 4, figsize = (28,7)) #gridspec_kw={'width_ratios': [1,2]})
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    for j in range(len(axs)):
        axs[j].tick_params(axis='x', labelsize= 23)
        axs[j].tick_params(axis='y', labelsize= 23)

    for task in ["Gender_F", "Gender_M", "Age"]:
        table1[task] = {}
        for idx, split in enumerate(["all", "train", "val", "test"]):
            data = results[split]
            subdata = data.drop_duplicates(subset='pat_name_id', keep="first")
            data = subdata.copy()
            # could use pat_ckey as well,
            #len(set(data["pat_ckey"]))
            #len(set(data["pat_name_id"]))

            namemap = {"all": "All", "train": "Training", "val": "Validation", "test": "Test"}
            colname = namemap[split]
            if task == "Gender_F" or task == "Gender_M":
                gd = data["Sex"] == task.split("_")[1]
                gd = gd[gd == True]
                v = int(round(100*len(gd)/data.shape[0], 0))
                v = str(v) + " % (" + str(len(gd))
                v = v + "/ " + str( data.shape[0] ) +")"
                table1[task].update({colname: v })
            else:
                v = str (np.round(np.mean(data[task]),1) )
                v = v + " +/- " + str (np.round(np.std(data[task]),1 ))
                v = v + " (N = " + str( data.shape[0] ) +")"
                table1[task].update({colname: v })
                axs [idx].hist (data[task], bins = 42)
                ylim = (0,110)
                axs[idx].set_ylim(ylim)
            if "Gender" not in task and split == "all":
                print (task, "range: ", np.min(data[task]), np.max(data[task]))

    table1 = pd.DataFrame.from_dict(table1, orient = "index")
    table1.to_excel("./paper/table1.xlsx")

    axs[0].set_xlabel("Age (All Patients)", fontsize = 30)
    axs[1].set_xlabel("Age (Training)", fontsize = 30)
    axs[2].set_xlabel("Age (Validation)", fontsize = 30)
    axs[3].set_xlabel("Age (Test)", fontsize = 30)
    axs[0].set_ylabel("Number of Patients", fontsize = 30)

    f.suptitle("Age distribution ", fontsize = 32)
    plt.tight_layout()

    f.savefig ("./paper/Figure_3.png", dpi = 200)
    pass


def plotResults (testResults, ttset = None, fname = None):
    fTbl = testResults.copy()
    f, axs  = plt.subplots(1, 3, figsize = (32,9))
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    for j in range(len(axs)):
        axs[j].tick_params(axis='x', labelsize= 23)
        axs[j].tick_params(axis='y', labelsize= 23)
    np.random.seed(42)
    random.seed(42)

    fTbl["error"] = fTbl["Age"] - fTbl["Predictions"]

    # plot them
    colors = {'F':'red', 'M':'blue'}
    axs [0].scatter (fTbl["age"], fTbl["Predictions"], color = fTbl["Sex"].map(colors))
    vmin = np.min([fTbl["age"], fTbl["Predictions"]])
    vmax = np.max([fTbl["age"], fTbl["Predictions"]])

    axs[0].plot([vmin, vmax], [vmin, vmax], '-', color = 'k')
    axs[0].plot([vmin+4, vmax+4], [vmin, vmax], linestyle = 'dashed', color = 'k')
    axs[0].plot([vmin-4, vmax-4], [vmin, vmax], linestyle = 'dashed', color = 'k')
    axs[0].set_xlim([0, 21.5])


    diffs = fTbl["Predictions"]-fTbl["age"]
    axs [1].hist (diffs, bins = 42)
    mae = np.mean(np.abs (fTbl["age"] -fTbl["Predictions"]))
    maeStd = np.std(np.abs(fTbl["age"] -fTbl["Predictions"]))
    print ("MAE:", mae, "+/-",  maeStd)
    print ("ME shift:", np.sum (fTbl["age"] -fTbl["Predictions"])/fTbl.shape[0])
    axs[1].set_ylim([0,72])
    axs[1].set_xlim([-8,8])

    # make bland-altman plot thing
    fTbl["diffs"] = fTbl["age"] -fTbl["Predictions"]

    axs[2].scatter(fTbl["age"], fTbl["diffs"], color = fTbl["Sex"].map(colors))
    axs[2].plot([0, 21.0], [0, 0], '-', color = 'k')
    axs[2].set_ylim([-8, 8])


    axs[0].set_xlabel ("Ground truth age [y]", fontsize = 30)
    axs[0].set_ylabel ("Predicted age [y]", fontsize = 30)

    axs[1].set_xlabel ("Prediction difference [y]", fontsize = 30)
    axs[1].set_ylabel ("Count", fontsize = 30)

    axs[2].set_xlabel ("Ground truth age [y]", fontsize = 30)
    axs[2].set_ylabel ("Prediction error [y]", fontsize = 30)

    f.suptitle("Results on the " + ttset, fontsize = 32)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.4)
    f.savefig ("./paper/" + fname, dpi = 300)


    fTbl["error"] = fTbl["age"] - fTbl["Predictions"]
    examples = fTbl[ abs(fTbl["error"]) > 2.0]
    print (len(examples), " are beyond 2.0, from ", fTbl.shape[0], " scans")
    print ("this is ", len(examples)/fTbl.shape[0], " %")
    examples = fTbl[ abs(fTbl["error"]) > 4.0]
    print (len(examples), " are beyond 2.0 , from ", fTbl.shape[0], " scans")
    print ("this is ", len(examples)/fTbl.shape[0], " %")




def getPreds (model, split = "test"):
    # reread val sets and recompute MAE
    vTbl = pd.read_csv("./results/" + model + "." + split + ".csv")
    vTbl[split+"_preds"] = vTbl["preds"]
    diffs = np.abs(vTbl[split+"_preds"] - vTbl["Age"])
    MAE = np.mean(diffs)
    Std = np.std(diffs)
    print ("MAE of", model, "is", MAE, "+/-", Std)
    Err = diffs > 2
    print ("Err@2y of", model, "is", np.sum(Err), "/", len(diffs), ":", np.sum(Err)/len(diffs))
    Err = diffs > 4
    print ("Err@4y of", model, "is", np.sum(Err), "/", len(diffs), ":", np.sum(Err)/len(diffs))
    cor = scipy.stats.pearsonr(vTbl["Age"], vTbl[split+"_preds"])
    print ("Correlation of", model, "is", cor)
    return MAE, vTbl


def identifyBestModel ():
    # reread val sets and recompute MAE
    results = {}
    for model in ["AMR", "Coral", "Corn", "StandardMeta", "Standard"]:
        MAE, vTbl = getPreds (model, "validation")
        results[model] = MAE
    return results


def findExamples (testResults, ttset = None):
    fTbl = testResults.copy()
    fTbl["diffs"] = np.abs (fTbl["age"] -fTbl["Predictions"])
    fTbl = fTbl.sort_values(["diffs"])

    minE = fTbl.iloc[0]
    maxE = fTbl.iloc[-1]
    medianE = fTbl.iloc[len(fTbl)//2]
    print ("Min", minE["age"], minE["Predictions"], " D:", minE["diffs"])
    print ("Median", medianE["age"], medianE["Predictions"], " D:", medianE["diffs"])
    print ("Max", maxE["age"], maxE["Predictions"], " D:", maxE["diffs"])

    finalImg = np.zeros ((512+2*16, 2*512+3*16, 3), dtype = np.uint8)
    finalImg = finalImg*0 + 255
    k = cv2.imread("./images/"+minE.Image)
    k = cropImage(k, crop = (248,512))
    k [200:248,:,:] = 0
    age = np.round(minE["age"], 1)
    pred = np.round(minE["Predictions"], 1)
    k = addText (k, text="True: "+str(age) + "y, Prediction: " + str(pred) +"y", org = (20, 215), fontSize = 24)
    k = addText (k, text="(A)", org = (470, 215), fontSize = 24)
    finalImg [16:16+248, 16:512+16, :] = k[0:248, :, :]

    k = cv2.imread("./images/"+medianE.Image)
    k = cropImage(k, crop = (248,512))
    age = np.round(medianE["age"], 1)
    pred = np.round(medianE["Predictions"], 1)
    k = addText (k, text="True: "+str(age) + "y, Prediction: " + str(pred) +"y", org = (20, 215), fontSize = 24)
    k = addText (k, text="(B)", org = (470, 215), fontSize = 24)
    finalImg [8+256+16:8+16+256+248, 16:512+16, :] =  k[0:248, :, :]


    k = cv2.imread("./images/"+maxE.Image)
    k = cropImage(k, crop = (264,264), ofs = (170,120))
    k = cv2.resize(k, (512, 512))
    k [474:512,:,:] = 0
    age = np.round(maxE["age"], 1)
    pred = np.round(maxE["Predictions"], 1)
    k = addText (k, text="True: "+str(age) + "y, Prediction: " + str(pred) +"y", org = (20, 480), fontSize = 24)
    k = addText (k, text="(C)", org = (470, 480), fontSize = 24)
    finalImg [16:16+512, 16*2+512*1:16*2+512*2, :] = k


    finalImg = cv2.resize(finalImg, (0,0), fx = 2.5, fy = 2.5)

    cv2.imwrite("paper/Figure_7.png", finalImg)
    pass



def plotSingleCurve (results, axs = None, model = None):
    if model is None:
        model = results["model"]
    log_dir = "./checkpoints/" + model + "/best/logs/version_0"
    reader = SummaryReader(log_dir)
    df = reader.scalars

    train_loss = df.query("tag == 'train_loss'")
    valid_loss = df.query("tag == 'valid_loss'")
    train_mae = df.query("tag == 'train_mae'")
    valid_mae = df.query("tag == 'valid_mae'")

    axs.plot(train_loss["step"], train_loss["value"], label='Training', c='b')
    axs.plot(valid_loss["step"], valid_loss["value"], label='Validation', c='g')
    axs.legend(loc="upper right", prop={'size': 23})

    # for j in range(len(axs)):
    axs.tick_params(axis='x', labelsize= 21)
    axs.tick_params(axis='y', labelsize= 21)
    trl = {"AMR": "AMR", "Coral": "CORAL", "Corn": "CORN", "StandardMeta": "L1+DICOM-Tags", "Standard": "L1"}

    axs.set_title("Loss curves for "+trl[model], fontsize = 30)
    axs.set_xlabel('Step', fontsize=25)
    axs.set_ylabel('Loss', fontsize=25)   # relative to plt.rcParams['font.size']
    pass


def plotCurves (results):
    f, axs  = plt.subplots(2, 3, figsize = (26,16)) #gridspec_kw={'width_ratios': [1,2]})
    for j, model in enumerate(["AMR", "Coral", "Corn", "StandardMeta", "Standard"]):
        ix = j//3
        iy = j%3
        plotSingleCurve(results, axs[ix][iy], model)
    f.tight_layout(pad = 5.5)
    f.delaxes(axs[1][2])
    f.savefig ("./paper/Supp_Figure_1.png", dpi = 200)


def printModelParameters (results):
    for m in results.keys():
        print ("\n###", m)
        tResults = pd.read_csv("./results/optuna.results."+m+".csv")
        bestModel = tResults[tResults.value == np.min(tResults.value)].iloc[0]
        best_params = {k.replace("params_", ''):bestModel[k] for k in bestModel.keys() if "params_" in k}
        pprint (best_params)


if __name__ == "__main__":
    results = identifyBestModel()
    bestModel = min(results, key=results.get)
    results = {"model": bestModel}

    # base data
    data = getData("train", "./data")
    results["train"] = data.copy()

    data = getData("val", "./data")
    MAE, vTbl = getPreds (bestModel, "validation")
    assert (len(data) == len(vTbl))
    data["Predictions"] = vTbl["validation_preds"]
    results["val"] = data.copy()

    data = getData("test", "./data")
    MAE, vTbl = getPreds (bestModel, "test")
    assert (len(data) == len(vTbl))
    data["Predictions"] = vTbl["test_preds"]
    results["test"] = data.copy()

    allData = pd.concat([results[s] for s in ["train", "test", "val"]], axis = 0)
    results["all"] = allData.copy()
    plotAgeHistogramm(results)

    generateTopo(results)

    plotResults (results["val"], ttset = "validation set", fname = "Figure_5.png")
    plotResults (results["test"], ttset = "test set", fname = "Figure_6.png")

    findExamples(results["test"])
    plotCurves (results)
    printModelParameters (results)
#
