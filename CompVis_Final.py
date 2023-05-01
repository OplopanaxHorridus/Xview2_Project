# from Final_Package.Count_Houses import Count_Houses
import cv2 as cv
import numpy as np
import math 

# Importing the libraries
from matplotlib import pyplot as plt


def adjustContrast(img_in, clip, tileGrid):
    # Convert to LAB color space
    img_lab = cv.cvtColor(img_in, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(img_lab)

    # Applying CLAHE to L-channel
    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=tileGrid)
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channels
    img_out = cv.merge((cl, a, b))

    # Converting image from LAB back to BGR
    img_out = cv.cvtColor(img_out, cv.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    # img_out = np.hstack((img_in, img_out))

    return img_out

def resizeImage(img_in, height, width):
    new_dimensions = (height, width)

    img_out = cv.resize(img_in,
                        new_dimensions,
                        interpolation = cv.INTER_LINEAR)

    return img_out
  
# Reading the image and converting into B/W
# img1 = cv2.imread('pre_disaster.jpg')
img1 = cv.imread(r"C:\Users\dylan\OneDrive\Documents\WPI\Computer_Vision\Photos\Test_image2.png")

img1 = adjustContrast(img1, 2.0, (8, 8))

# img1_gray = cv2.GaussianBlur(img1_gray, (7, 7),0)
img1 = cv.medianBlur(img1, 11)
myimage= resizeImage(img1, 512, 512)
cv.imshow("Enhanced Image", resizeImage(img1, 512, 512))
cv.waitKey(0)

img1_gray = cv.cvtColor(myimage, cv.COLOR_BGR2GRAY)

# Harris Corner Detector
img1_gray = np.float32(img1_gray)
blkSize = 2
kSize = 3
k = 0.04
dest = cv.cornerHarris(img1_gray, blkSize, kSize, k)

# result is dilated for marking the corners, not important
dest = cv.dilate(dest, None)

# Threshold for an optimal value, it may vary depending on the image
img_out = img1
img_out[dest > (0.01 * dest.max())] = [0, 0, 255]

print(img_out)
# # ORB (SIFT+SURF) function
# orb = cv.ORB_create()
# keypnt1, dest1 = orb.detectAndCompute(img1_gray, None)

# # Drawing the keypoints (ORB; flags=0 for default)
# kp = orb.detect(img1_gray, None)
# kp, desc = orb.compute(img1_gray, kp)
# kp_image = cv.drawKeypoints(img1, kp, None, color=(0, 255, 0), flags=0)

# img_orig = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
# # img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
# kp_out = cv.cvtColor(kp_image, cv.COLOR_BGR2RGB)

#May need it to go through a folder as a list and go through each image in the folder up here

# img = cv.imread(r"C:\Users\dylan\OneDrive\Documents\WPI\Computer_Vision\Photos\Test_Two.png")

#count the amount of houses

def CountHouses(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(image=thresh_img, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    #draw_contours = contours
    # print(contours)
    cv.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    cv.drawContours(image=gray_img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    print(len(contours))
    img = cv.resize(img,[256,256])
    cv.imshow("test", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

CountHouses(img_out)



#img = cv.imread(r"C:\Users\dylan\OneDrive\Documents\WPI\Computer_Vision\Computer_Vision\Count_Houses_test.png")
#img = cv.imread(r'C:\Users\dylan\OneDrive\Documents\WPI\Computer_Vision\Photos\Project_CS549_1.jpg')
# CountHouses(kp_out)

#img = cv.imread(r"C:\Users\dylan\OneDrive\Documents\WPI\Computer_Vision\Computer_Vision\Count_Houses_test.png")
#img = cv.imread(r'C:\Users\dylan\OneDrive\Documents\WPI\Computer_Vision\Photos\Project_CS549_1.jpg')






