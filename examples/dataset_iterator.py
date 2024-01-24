from stereo_pcd.datasets.middlebury_dataset import MiddleburyDataset

if __name__ == "__main__":
    dataset = MiddleburyDataset("...")

    for sample_name, sample in dataset:
        print(sample_name)
