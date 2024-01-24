import numpy as np
from PIL import Image

from stereo_pcd.geometry import (
    calc_focal,
    calc_projection_matrix,
    pixel_to_world,
    world_to_pixel,
)
from stereo_pcd.simple_stereo import (
    calc_disparity_map,
    create_disparity_picture,
)


def main() -> None:
    left_array = np.zeros((600, 800, 4), np.uint8)
    right_array = np.zeros((600, 800, 4), np.uint8)
    left_array[:, :, 3].fill(255)

    left_array[424][587][0:3] = np.array([255, 255, 255])
    right_array[424][676][0:3] = np.array([255, 255, 255])

    im_height, im_width, _ = left_array.shape
    fov = 90
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

    print(f"{focal=}")
    print(pixel_to_world(projection_cam1, 630, 468, 5.5))
    print(pixel_to_world(projection_cam2, 676, 424, 4.5))

    print(world_to_pixel(projection_cam1, -6.9, -10.6, 27.5))
    print(world_to_pixel(projection_cam2, -6.9, -10.6, 27.5))

    print(world_to_pixel(projection_cam1, -6.4, -10.6, 27.5))

    disparity_bitmap = create_disparity_picture(
        calc_disparity_map(left_array, right_array, 0, 100)
    )
    Image.fromarray(disparity_bitmap).show()


if __name__ == "__main__":
    main()
