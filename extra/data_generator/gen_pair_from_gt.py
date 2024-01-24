import cv2 as cv
import numpy as np
import random


def is_black(pixel):
    return np.mean(pixel) == 0


def random_color():
    return np.array([random.randint(0, 255) for _ in range(3)], dtype=np.uint8)


gt = cv.imread("gt3.png", cv.IMREAD_GRAYSCALE)
img = cv.imread("img3.png", cv.IMREAD_COLOR)


left = np.zeros((*gt.shape, 3), dtype=np.uint8)
right = np.zeros((*gt.shape, 3), dtype=np.uint8)

for i in range(gt.shape[0]):
    for j in range(gt.shape[1]):
        if not is_black(gt[i][j]):
            color = img[i][j]  # random_color() #[255, 0 , 0]
            left[i][j] = color
            right[i][j + gt[i][j]] = color

# Fill gaps in right image

for i in range(right.shape[0]):
    try:
        start_filling = next(
            j for j in range(right.shape[1]) if not is_black(right[i][j])
        )
        end_filling = next(
            j for j in range(right.shape[1] - 1, -1, -1) if not is_black(right[i][j])
        )
    except StopIteration:
        continue

    for j in range(start_filling, end_filling):
        if is_black(right[i][j]):
            right[i][j] = right[i][j - 1]


cv.imwrite("left3.png", left)
cv.imwrite("right3.png", right)
