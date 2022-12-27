from torchvision import transforms as tf
import torchvision.transforms
import torch

NUM_CLASSES = 21*4
START_AGE = 0
END_AGE = 21*4


rootDir = "/data/data/age.ped.localizer/"
mean, std= torch.tensor([0.4359, 0.3436, 0.2514]), torch.tensor([0.2183, 0.2509, 0.2213])

def getTransform (split, imgSize):
    if imgSize == 224:
        imgPad = 16
        cropPad = 16
    if imgSize == 512:
        imgPad = 32
        cropPad = 32
    # else we will get an error
    transforms_train = torchvision.transforms.Compose([
        tf.ToPILImage(),
        tf.Resize((imgSize + imgPad, imgSize + imgPad)),
        tf.RandomRotation(degrees=(-10, 10)),
        tf.ColorJitter(brightness=0.1, contrast=0.10),
        tf.RandomCrop(imgSize, padding=(cropPad,cropPad,cropPad,1), padding_mode='constant'),
        tf.ToTensor(),
        tf.Normalize(mean, std)
    ])

    transforms_test = transforms_val = torchvision.transforms.Compose([
        tf.ToPILImage(),
        tf.Resize((imgSize, imgSize)),
        tf.ToTensor(),
        tf.Normalize(mean, std)
    ])

    if split == "train":
        return transforms_train
    elif split == "val" or split == "test" or split == "valid":
        return transforms_val
    pass


#
