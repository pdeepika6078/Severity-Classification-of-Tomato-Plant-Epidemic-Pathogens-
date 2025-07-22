import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def Image_Results():
    for i in range(1):
        Orig = np.load('img.npy', allow_pickle=True)
        segment = np.load('seg.npy', allow_pickle=True)
        ind = [0, 1, 2, 3, 4]
        for j in range(len(ind)):
            original = Orig[ind[j]]
            seg = segment[ind[j]]
            fig, ax = plt.subplots(1, 2)
            plt.suptitle('Images', fontsize=20)

            plt.subplot(1, 2, 1)
            plt.title('Orig')
            plt.imshow(original)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title('Segmentation')
            plt.imshow(seg)
            plt.axis('off')

            plt.show()
            cv.imwrite('./Results/Bacterial_spot-' + str(i + 1) + 'orig-' + str(j + 1) + '.png', original)
            cv.imwrite('./Results/Bacterial_spot-' + str(i + 1) + 'segment-' + str(j + 1) + '.png', seg)


def Early_Image_Results():
    for i in range(1):
        Orig = np.load('Early_img.npy', allow_pickle=True)
        segment = np.load('Early_seg.npy', allow_pickle=True)
        ind = [0, 1, 2, 3, 4]
        for j in range(len(ind)):
            original = Orig[ind[j]]
            seg = segment[ind[j]]
            fig, ax = plt.subplots(1, 2)
            plt.suptitle('Images', fontsize=20)

            plt.subplot(1, 2, 1)
            plt.title('Orig')
            plt.imshow(original)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title('Segmentation')
            plt.imshow(seg)
            plt.axis('off')

            plt.show()
            cv.imwrite('./Results/Early-' + str(i + 1) + 'orig-' + str(j + 1) + '.png', original)
            cv.imwrite('./Results/Early-' + str(i + 1) + 'segment-' + str(j + 1) + '.png', seg)

def Septoria_leaf_spot_Image_Results():
    for i in range(1):
        Orig = np.load('Septoria_leaf_imge.npy', allow_pickle=True)
        segment = np.load('Septoria_leaf_seg.npy', allow_pickle=True)
        ind = [0, 1, 2, 3, 4]
        for j in range(len(ind)):
            original = Orig[ind[j]]
            seg = segment[ind[j]]
            fig, ax = plt.subplots(1, 2)
            plt.suptitle('Images', fontsize=20)

            plt.subplot(1, 2, 1)
            plt.title('Orig')
            plt.imshow(original)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title('Segmentation')
            plt.imshow(seg)
            plt.axis('off')

            plt.show()
            cv.imwrite('./Results/Septoria_leaf_spot-' + str(i + 1) + 'orig-' + str(j + 1) + '.png', original)
            cv.imwrite('./Results/Septoria_leaf_spot-' + str(i + 1) + 'segment-' + str(j + 1) + '.png', seg)



if __name__ == '__main__':
    Image_Results()
    Early_Image_Results()
    Septoria_leaf_spot_Image_Results()

