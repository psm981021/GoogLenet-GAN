import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
import urllib
# from googlenet import GoogLeNet 


import torchvision.models as models


def random_plot(batch_images):
    import matplotlib.pyplot as plt	
    import random
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    batch_images = batch_images
    batch_size = batch_images.size(0)
    num = random.randrange(0,batch_size)

    n_features = random.sample(range(batch_size), 16)
    fig = plt.figure()
    cols = 4
    rows = 4

    i = 1
    for idx in n_features:
        img = batch_images[idx].detach().numpy()
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([]), ax.set_yticks([])
        i += 1

    plt.show()

"""디버깅용"""
if __name__ == '__main__':

    model = models.googlenet(pretrained = True)
    # model = GoogLeNet()
    model.load_state_dict(torch.load("./model/googlenet_pretrained.pt", map_location=torch.device('cpu')), strict=False, )
    model.eval()

    input_image = Image.open("./dataset/cat.jpg")
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    input_tensor = preprocess(input_image)
    input_tensor = input_tensor.reshape(1, 3, 224, 224)
    output, feature = model(input_tensor)
    print(feature.size())
    pred = torch.argmax(output, dim=1)
    pred = [p.item() for p in pred]
    if pred[0] == 0:
        print("cat")
    else:
        print("dog")

# %%
# input_image = Image.open("./dataset/dog.jpg")
# preprocess = transforms.Compose([
#     transforms.Resize((128,128)),
#     transforms.ToTensor()
# ])
# input_tensor = preprocess(input_image)
# input_tensor = input_tensor.reshape(1, 3, 128, 128)
# output = model(input_tensor)
# pred = torch.argmax(output, dim=1)
# pred = [p.item() for p in pred]
# if pred[0] == 0:
#     print("cat")
# else:
#     print("dog")


         



