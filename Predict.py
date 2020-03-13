import os
from torchvision import transforms

def Predict(datapath, stylefile, testfile, model):
    style = os.join(datapath, 'style', stylefile)
    test = os.join(datapath, 'test', testfile)
    transform = transforms.Compose([transforms.Resize((500, 500)),
         transforms.ToTensor()])

    test = transform(test)

    result = model(style, test)
    result.save()