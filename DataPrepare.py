import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class InputData(Dataset):
    """
    Style & Sketch
    """
    def __init__(self, datadir='./data', mode='train',transform=None, colorhisto=True, size=500):
        """
        :param datadir: data directory
        :param mode: train / validate
        :param transform: process image
        :param colorhisto: whether need to caculate color histogram
        :param size: image size limitation
        """
        datadir = datadir + '/' + mode
        self.datadir = datadir
        self.mode = mode
        self.transform = transform
        self.colorhisto = colorhisto
        self.size = size
        self.filelist = []
        for i in os.listdir(datadir):
            if os.path.splitext(i)[-1] == ".png":
                self.filelist.append(i)

    def getData(self, index):
        """
        :param index: file index in the filelist
        :return: Original image, Sketch, ColorHistogram
        """
        file = self.filelist[index]
        filepath = self.datadir + '/' + file
        filename = os.path.splitext(file)[0]


        ColorHistogram = None
        if self.colorhisto:
            with open(('./data/colorgram/' + filename +'.json'),'r') as colorhisto_file:
                colorh = json.loads(colorhisto_file.read())
            # Make colorgram to sensor
            colors = list(colorh.values())
            topnum = len(colors[0].keys())
            tensor = np.ones([topnum*3, self.size, self.size])

            h = self.size // 4
            for i in range(len(colors)):
                hi = h * i
                color = colors[i]
                for j in range(1, topnum + 1):
                    r, g, b = color[str(j)]

                    # assign index
                    red = (j - 1) * 3
                    green = (j - 1) * 3 + 1
                    blue = (j - 1) * 3 + 2

                    # assign values
                    tensor[red, hi:hi + h] *= r
                    tensor[green, hi:hi + h] *= g
                    tensor[blue, hi:hi + h] *= b

            tensor = torch.from_numpy(tensor.copy())
            # TODO: Scale?
            ColorHistogram = (tensor / 255.)

        # Get Original Image and Sketch
        image = Image.open(filepath)
        width, height = image.size
        OriginalImage =image.crop((0,0,width//2,height))
        SketchImage = image.crop((width//2,0,width,height))

        # Resize
        tmpwidth_pad = max(self.size - width // 2, 0) // 2 + 1
        tmpheight_pad = max(self.size - height, 0) // 2 + 1

        ResizeImage = transforms.Compose([
            # Left & Right, Top & Bottom
            transforms.Pad((tmpwidth_pad, tmpheight_pad), 255),
            transforms.CenterCrop(self.size),
        ])

        OriginalImage = ResizeImage(OriginalImage)
        SketchImage = ResizeImage(SketchImage)

        # Transform Image
        if self.transform:
            OriginalImage = self.transform(OriginalImage)
            SketchImage = self.transform(SketchImage)

        # Todo: Why scale?

        return OriginalImage, SketchImage, ColorHistogram









