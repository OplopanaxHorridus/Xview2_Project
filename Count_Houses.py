import cv2 as cv
import numpy as np
import math 



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
#img = cv.imread(r"C:\Users\dylan\OneDrive\Documents\WPI\Computer_Vision\Computer_Vision\Count_Houses_test.png")
#img = cv.imread(r'C:\Users\dylan\OneDrive\Documents\WPI\Computer_Vision\Photos\Project_CS549_1.jpg')
img = cv.imread(r"C:\Users\dylan\OneDrive\Documents\WPI\Computer_Vision\Photos\Test_Two.png")
CountHouses(img)
