import json
import os
from PIL import Image
import cv2

def CaculateColorHisto(datapath):
    datalist = os.listdir(datapath + '/'+ 'train')
    for item in datalist:
        out_file = os.path.join(datapath,'colorhisto', '%s.json' % item)
        if os.path.exists(out_file):
            # for continuation
            print('%s already done!' % item)
            continue
        print('processing %s...' % item)

        imageHisto = {}
        image = cv2.imread(datapath + '/'+ 'train' + '/' + item, 0)
        [width, height] = image.shape[:2]
        for i in range(1,5):
            if i == 1:
                Roi=image[0:width//2, 0:height//2]
            elif i == 2:
                Roi = image[width // 2 : width, 0:height // 2]
            elif i==3:
                Roi = image[0:width // 2, height // 2 : height]
            else: #i == 4:
                Roi = image[width // 2 : width, height // 2 : height]

            histb = cv2.calcHist([image],[0],Roi, [256], [0, 255])
            histg = cv2.calcHist([image],[1],Roi, [256], [0, 255])
            histr = cv2.calcHist([image],[2],Roi, [256], [0, 255])
            histb.sort()
            histg.sort()
            histr.sort()
            imageHisto[i] = {1:[histr[0], histg[0], histb[0]],2:[histr[1], histg[1], histb[1]],3:[histr[2], histg[2], histb[2]],4:[histr[3], histg[3], histb[3]]}
            with open(out_file, 'w') as outFile:
                json.write(json.dumps(imageHisto))

CaculateColorHisto('./tmp')