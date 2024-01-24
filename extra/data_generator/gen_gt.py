import cv2 as cv
import numpy as np
from stereo_pcd.geometry import world_to_pixel, calc_projection_matrix, calc_focal


def blue(x, y):
    return np.array([255, 0, 0], dtype=np.uint8)


def green(x, y):
    return np.array([0, 255, 0], dtype=np.uint8)


def red(x, y):
    return np.array([0, 0, 255], dtype=np.uint8)


im_height, im_width = 600, 800
fov = 90

gt = np.zeros((im_height, im_width), np.uint8)
img = np.zeros((im_height, im_width, 3), np.uint8)

focal = calc_focal(fov, im_width)

projection_cam1 = calc_projection_matrix(
    fov,
    im_height,
    im_width,
    np.array([0, 0, 0], dtype=np.float32),
    np.array([9, 12, -23], dtype=np.float32),
)
projection_cam2 = calc_projection_matrix(
    fov,
    im_height,
    im_width,
    np.array([0, 0, 0], dtype=np.float32),
    np.array([10, 12, -23], dtype=np.float32),
)

cube_center = (-6.9, -10.6, 28)
# cube_center = (-12.1, -10.6, 28)
side_lenght = 1

visible = ["front", "top", "left"]

x_min = cube_center[0] - side_lenght / 2
x_max = cube_center[0] + side_lenght / 2

y_min = cube_center[1] - side_lenght / 2
y_max = cube_center[1] + side_lenght / 2

z_min = cube_center[2] - side_lenght / 2
z_max = cube_center[2] + side_lenght / 2

z_edge = z_min if "front" in visible else z_max
y_edge = y_min if "top" in visible else y_max
x_edge = x_min if "left" in visible else x_max

ths = 0.1
lin_num = 2500

gt_cam = projection_cam1
second_cam = projection_cam2

# przednia ściana
for x in np.linspace(x_min, x_max, num=lin_num):
    for y in np.linspace(y_min, y_max, num=lin_num):
        [px, py] = world_to_pixel(gt_cam, x, y, z_edge)
        if px - int(px) < ths and py - int(py) < ths:
            if gt[int(py)][int(px)] == 0:
                [px2, py2] = world_to_pixel(second_cam, x, y, z_edge)
                gt[int(py)][int(px)] = abs(round(px2 - px))
                img[int(py)][int(px)] = blue(0, 0)


# górna ściana
for x in np.linspace(x_min, x_max, num=lin_num):
    for z in np.linspace(z_min, z_max, num=lin_num):
        [px, py] = world_to_pixel(gt_cam, x, y_edge, z)
        if px - int(px) < ths and py - int(py) < ths:
            if gt[int(py)][int(px)] == 0:
                [px2, py2] = world_to_pixel(second_cam, x, y_edge, z)
                gt[int(py)][int(px)] = abs(round(px2 - px))
                img[int(py)][int(px)] = green(0, 0)

# lewa ściana
for y in np.linspace(y_min, y_max, num=lin_num):
    for z in np.linspace(z_min, z_max, num=lin_num):
        [px, py] = world_to_pixel(gt_cam, x_edge, y, z)
        if px - int(px) < ths and py - int(py) < ths:
            if gt[int(py)][int(px)] == 0:
                [px2, py2] = world_to_pixel(second_cam, x_edge, y, z)
                gt[int(py)][int(px)] = abs(round(px2 - px))
                img[int(py)][int(px)] = red(0, 0)


cv.imwrite("out3.png", gt)
cv.imwrite("img3.png", img)
