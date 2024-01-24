import cv2 as cv
import numpy as np
from stereo_pcd.geometry import world_to_pixel, calc_projection_matrix, calc_focal

im_height, im_width = 600, 800
fov = 90

left_array = np.zeros((im_height, im_width), np.uint8)
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

# przednia ściana
for x in np.linspace(-7.4, -6.4, num=2500):
    for y in np.linspace(-11.1, -10.1, num=2500):
        [px, py] = world_to_pixel(projection_cam1, x, y, 27.5)
        if px - int(px) < 0.1 and py - int(py) < 0.1:
            if left_array[int(py)][int(px)] == 0:
                [px2, py2] = world_to_pixel(projection_cam2, x, y, 27.5)
                left_array[int(py)][int(px)] = round(px2 - px)


# górna ściana
for x in np.linspace(-7.4, -6.4, num=2500):
    for z in np.linspace(27.5, 28.5, num=2500):
        [px, py] = world_to_pixel(projection_cam1, x, -11.1, z)
        if px - int(px) < 0.1 and py - int(py) < 0.1:
            if left_array[int(py)][int(px)] == 0:
                [px2, py2] = world_to_pixel(projection_cam2, x, -11.1, z)
                left_array[int(py)][int(px)] = round(px2 - px)

# lewa ściana
for y in np.linspace(-11.1, -10.1, num=2500):
    for z in np.linspace(27.5, 28.5, num=2500):
        [px, py] = world_to_pixel(projection_cam1, -7.4, y, z)
        if px - int(px) < 0.1 and py - int(py) < 0.1:
            if left_array[int(py)][int(px)] == 0:
                [px2, py2] = world_to_pixel(projection_cam2, -7.4, y, z)
                left_array[int(py)][int(px)] = round(px2 - px)


cv.imwrite("out.png", left_array)
