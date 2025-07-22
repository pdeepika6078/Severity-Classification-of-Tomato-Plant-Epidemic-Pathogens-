import cv2
import matplotlib.pyplot as plt
import numpy as np


def Make_Image_Results():
    img =[]
    Images = np.load('Images.npy', allow_pickle=True)
    # GT = np.load('Seg_Img_' + str(n + 1) + '.npy', allow_pickle=True)
    for i in range(len(Images)):
        print([i])
        image = Images[i]
        segm = image.copy()
        im = cv2.rectangle(segm, (200, 100), (350, 500), (0, 255, 255), 4)
        # cv2.rectangle(segm, (100, 300), (200, 400), (0, 255, 0), 2)
        img.append(im)
    np.save('Segmentation.npy', img)



# def Make_Image_Results():
#     no_of_dataset = 2
#     IMAGE = [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4]]
#     for n in range(no_of_dataset):
#         Images = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)
#         GT = np.load('Seg_Img_' + str(n + 1) + '.npy', allow_pickle=True)
#         for i in range(len(Images[IMAGE[n]])):
#             print(IMAGE[n][i])
#             image = Images[IMAGE[n][i]]
#             seg = GT[IMAGE[n][i]]
#             segm = image.copy()
#             cv2.rectangle(segm, (200, 100), (350, 500), (0, 255, 255), 4)
#             # cv2.rectangle(segm, (100, 300), (200, 400), (0, 255, 0), 2)
#             plt.subplot(1, 2, 1)
#             plt.title('Original Image')
#             plt.imshow(image)
#             plt.subplot(1, 2, 2)
#             plt.title('Predicted Image')
#             plt.imshow(segm)
#             # cv2.imwrite('./Results/Image_Results/Dataset' + str(n + 1) + '-Seg-' + str(i + 1) + '.png', segm)
#             plt.show()


if __name__ == '__main__':
    Make_Image_Results()
