import os
import torch
import re
import cv2
import numpy as np

import Binary_Food_Fruit_Classification_test
import MulitClass_Fruit_Classification_test
import Fruit_Binary_Segmentation_test
#Image reading 
# Integrated test processing
device = 'cuda' if torch.cuda.is_available() else 'cpu'
baseIntegratedTestPath='Test Cases Structure/Integerated Test/'
testFiles=os.listdir(baseIntegratedTestPath)

def main():
    for image in testFiles:
        imgPath = os.path.join(baseIntegratedTestPath, image)
        # remove the extension, then the file name , then the 'g'
        grams=float(image.split('.')[0].split('_')[1][:-1])
        
        if not os.path.exists(imgPath[:-4]):
            os.makedirs(imgPath[:-4])

        if(Binary_Food_Fruit_Classification_test.predict_image(imgPath)=='Food'):
            # get food type
            pass
            type=''
            # calculate no of calories
            calories=CalculateCalories('Food',type,grams)
            with open(os.path.join(imgPath[:-4],imgPath[:-4]+".txt"),'w') as f:
                f.write('Food\n')
        else:
            # get fruit type
            type=MulitClass_Fruit_Classification_test.test_image(imgPath)
            # calculate no of calories
            calories=CalculateCalories('Fruit',type,grams)
            with open(os.path.join(imgPath[:-4],image[:-4]+".txt"),'w') as f:
                f.write('Fruit\n')

            # apply binary segmentation
            binaryMask=Fruit_Binary_Segmentation_test.predict(imgPath)
            mask = binaryMask.squeeze()  
            mask = (mask * 255)
            mask = mask.astype(np.uint8)
            cv2.imwrite(os.path.join(imgPath[:-4],"Binary-Mask.png"), mask)
            # apply multi-segmentation
        
        with open(os.path.join(imgPath[:-4],image[:-4]+".txt"),'a') as f:
            f.write(f'{type}\n')
            f.write(f'{calories}\n')

def CalculateCalories(flag,className,grams):
    if(flag=='Fruit'):
        fileNames=['Project Data/Fruit/Calories.txt']
    else:
        fileNames=[
            'Project Data/Food/Train Calories.txt',
            'Project Data/Food/Val Calories.txt'
        ]

    for fileName in fileNames:
        with open(fileName,'r') as f:
            for line in f:
                if(line):
                    if(className.lower() in line.lower()):
                        caloriesPerGram = float(re.search(r"(\d+\.?\d*)", line).group(1))
                        return caloriesPerGram * grams

main()