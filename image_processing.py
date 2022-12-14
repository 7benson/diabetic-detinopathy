import cv2
import os
import numpy as np
from skimage import morphology

def extractBloodVessels(imageName):
    if(os.path.isfile(imageName)):
        # read the image in numpy format
        image = np.array(cv2.imread(imageName, 1))

        # get the green layer  out of the 3 RGB components of the image
        green_layer = image[:, :, 1]

        # applying histogram equalization on green layer
        histogram_equalised_img = cv2.equalizeHist(green_layer)

        # applying kirsch filter
        kirsch_filter_output = kirschFilter(histogram_equalised_img)

        # applying inv thresholding
        ret, inv_thresh_img = cv2.threshold(kirsch_filter_output, 160, 180, cv2.THRESH_BINARY_INV)

        # apply median filter
        final_output = morphology.remove_small_objects(inv_thresh_img, min_size=130, connectivity=100)

        cv2.imwrite('./static/output/extracted-vessels.png',final_output)
        print("Blood vessels have been successfully extracted into extracted-vessels.png")
    else:
        print("Input Image doesn't exist")



def kirschFilter(image):
    gray = image
    kernelG1 = np.array([[5, 5, 5],
                         [-3, 0, -3],
                         [-3, -3, -3]], dtype=np.float32)
    kernelG2 = np.array([[5, 5, -3],
                         [5, 0, -3],
                         [-3, -3, -3]], dtype=np.float32)
    kernelG3 = np.array([[5, -3, -3],
                         [5, 0, -3],
                         [5, -3, -3]], dtype=np.float32)
    kernelG4 = np.array([[-3, -3, -3],
                         [5, 0, -3],
                         [5, 5, -3]], dtype=np.float32)
    kernelG5 = np.array([[-3, -3, -3],
                         [-3, 0, -3],
                         [5, 5, 5]], dtype=np.float32)
    kernelG6 = np.array([[-3, -3, -3],
                         [-3, 0, 5],
                         [-3, 5, 5]], dtype=np.float32)
    kernelG7 = np.array([[-3, -3, 5],
                         [-3, 0, 5],
                         [-3, -3, 5]], dtype=np.float32)
    kernelG8 = np.array([[-3, 5, 5],
                         [-3, 0, 5],
                         [-3, -3, -3]], dtype=np.float32)
    g1 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g2 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g3 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG3), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g4 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG4), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g5 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG5), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g6 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG6), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g7 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG7), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g8 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG8), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    magn = cv2.max(g1, cv2.max(g2, cv2.max(g3, cv2.max(g4, cv2.max(g5, cv2.max(g6, cv2.max(g7, g8)))))))
    return magn

def extractExudates(imageName):
    if(os.path.isfile(imageName)):

        image = np.array(cv2.imread(imageName, 1))

        # get only the green component
        green_layer = image[:, :, 1]

        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE()
        clImg = clahe.apply(green_layer)

        strEl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        dilateImg = cv2.dilate(clImg, strEl)

        _, threshImg = cv2.threshold(dilateImg, 220, 220, cv2.THRESH_BINARY)

        medianImg = cv2.medianBlur(threshImg, 5)

        cv2.imwrite('./static/output/extracted-exudates.png', medianImg)
        print("Extraction of Exudates have been completed successfully")
    else:
        print("Input Image doesn't exist")