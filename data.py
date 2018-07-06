from __future__ import print_function, division
from os import listdir
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os.path import join
from PIL import Image

from skimage import io,color
import torch.utils.data as Data
import torchvision.transforms as transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

class perfect500k(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,image_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.root_dir = root_dir
        #self.transform = transform
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.transform = transform


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        input = io.imread(self.image_filenames[index])
        if self.transform:
            try:
                input = transforms.ToPILImage()(input)
                input = self.transform(input)
            except:
                print ("errr")



        return input,index
"""
datas=perfect500k('/Users/EASON/Shaofu/orz_data/perfect',transform= transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),

    ]))


dataloader = Data.DataLoader(datas, batch_size=1,
                        shuffle=False, num_workers=4)

for img,ad  in dataloader:
    try:
        img = img.numpy()
        img = img.reshape(3, 224, 224)
        img = img.transpose(1, 2, 0)
        img = Image.fromarray(np.uint8(img))
        path = "/Users/EASON/Shaofu/orz_data/perfect1/" + str(ad) + ".jpg"
        img.save(path)
        print(img)
        print(ad)
    except:
        print ("error")

"""
'''
fig = plt.figure()

for i in range(len(datas)):
    sample = datas[i]

    print(i)
    sample = transforms.ToPILImage()(sample).convert('RGB')
    plt.imshow(sample)
    plt.show()

'''