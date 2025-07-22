import os
import numpy as np
import pandas as pd
import cv2 as cv
from numpy import matlib
from EOA import EOA
from Global_Vars import Global_vars
from Model_AAMaskRCNN import Model_AAMaskRCNN
from Model_CNN import Model_CNN
from Model_DenseNet import Model_DenseNet
from Model_MobileNet import Model_MobileNet
from Model_PROPOSED import Model_PROPOSED
from Model_RESNET import Model_RESNET
from Obj_Cls import Objective_Function
from POA import POA
from PROPOSED import PROPOSED
from Plot_Results import *
from SGO import SGO
from TOA import TOA


def Read_Image(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    image = cv.resize(image, (512, 512))
    return image


def Read_Images(directory_name):
    Fold_Array = os.listdir(directory_name)
    Images = []
    Target = []
    iter = 1
    flag = 0
    for i in range(len(Fold_Array)):
        Img_Array = os.listdir(directory_name + Fold_Array[i])
        for j in range(len(Img_Array)):
            print(i, j)
            image = Read_Image(directory_name + Fold_Array[i] + '/' + Img_Array[j])
            Images.append(image)
            if Fold_Array[i][len(Fold_Array[i]) - 7:] == 'healthy':
                Target.append(0)
            else:
                flag = 1
                Target.append(iter)
        if flag == 1:
            iter = iter + 1
    return Images, Target


# Read Dataset
an = 0
if an == 1:
    Directory = './Datasets/'
    Dataset_List = os.listdir(Directory)
    for n in range(len(Dataset_List)):
        Images, Target = Read_Images(Directory + Dataset_List[n] + '/')
        np.save('Images.npy', Images)
        np.save('Target.npy', Target)

##Optimization for Segmentation
an = 0
if an == 1:
    Image = np.load('Images.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_vars.Image = Image
    Global_vars.Target = Target
    Npop = 10
    Ch_len = 3
    xmin = matlib.repmat(([5, 50, 300]), Npop, 1)
    xmax = matlib.repmat(([255, 100, 1000]), Npop, 1)
    initsol = np.zeros((xmax.shape))
    for p1 in range(Npop):
        for p2 in range(xmax.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    fname = Objective_Function
    Max_iter = 50

    print("EOA...")
    [bestfit1, fitness1, bestsol1, time1] = EOA(initsol, fname, xmin, xmax, Max_iter)

    print("SGO...")
    [bestfit2, fitness2, bestsol2, time2] = SGO(initsol, fname, xmin, xmax, Max_iter)

    print("TOA...")
    [bestfit3, fitness3, bestsol3, time3] = TOA(initsol, fname, xmin, xmax, Max_iter)

    print("POA...")
    [bestfit4, fitness4, bestsol4, time4] = POA(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

    best = ([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])

    np.save('Best_Sol.npy', best)

##Segmentation
an = 0
if an == 1:
    seg = []
    sol = np.load('Best_Sol.npy', allow_pickle=True)
    image = np.load('Images.npy', allow_pickle=True)
    for i in range(len(image)):
        Img = image[i]
        best_sol = sol.astype('int')
        Segmet = Model_AAMaskRCNN(Img, best_sol)
        seg.append(Segmet)
    np.save('Segmentation.npy', seg)

# Classification
an = 0
if an == 1:
    Feat = np.load('Segmentation.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Target = np.reshape(Target, (-1, 1))
    EVAL = []
    Activation = ['Linear', 'ReLU', 'TanH', 'Softmax', 'Sigmoid', 'Leaky ReLU']
    for learn in range(len(Activation)):
        Act = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:Act, :]
        Train_Target = Target[:Act, :]
        Test_Data = Feat[Act:, :]
        Test_Target = Target[Act:, :]
        Eval = np.zeros((5, 25))
        Eval[0, :] = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[1, :] = Model_RESNET(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[2, :] = Model_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[3, :] = Model_MobileNet(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[4, :] = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target)
        EVAL.append(Eval)
    np.save('Eval_ALL.npy', EVAL)

plot_Con_results()
ROC_curve()
Plot_Activation()
plot_results_Seg()
Image_Results()
Early_Image_Results()
Septoria_leaf_spot_Image_Results()
