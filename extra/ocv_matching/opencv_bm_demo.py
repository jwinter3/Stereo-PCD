import cv2 as cv
from matplotlib import pyplot as plt

# imgL = cv.imread("data/middlebury2001/scene1.row3.col1.png", cv.IMREAD_GRAYSCALE)
# imgR = cv.imread("data/middlebury2001/scene1.row3.col3.png", cv.IMREAD_GRAYSCALE)

# imgL = cv.imread("data/middlebury2021/im0.png", cv.IMREAD_GRAYSCALE)
# imgR = cv.imread("data/middlebury2021/im1.png", cv.IMREAD_GRAYSCALE)

imgL = cv.imread("000000_10.png", cv.IMREAD_UNCHANGED)
imgR = cv.imread("000000_10.png", cv.IMREAD_UNCHANGED)

stereo = cv.StereoBM_create(64, 9)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, "gray")
plt.savefig("disparity_BM.png")
