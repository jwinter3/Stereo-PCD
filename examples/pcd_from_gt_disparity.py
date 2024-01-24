from stereo_pcd.datasets.middlebury_dataset import MiddleburyDataset
from stereo_pcd.pointclouds import pseudo_points_color, save_txt


if __name__ == "__main__":
    dataset = MiddleburyDataset("...")

    sample = dataset.read_sample("ladder1")

    xyzrgb = pseudo_points_color(sample, sample.gt_disp)
    save_txt(xyzrgb, "ladder.txt")
