from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# NOTE: you need to download the Nordland dataset from  https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W
# this link is shared and maintained by the authors of VPR_Bench: https://github.com/MubarizZaffar/VPR-Bench
# the folders named ref and query should reside in DATASET_ROOT path
# I hardcoded the image names and ground truth for faster evaluation
# performance is exactly the same as if you use VPR-Bench.
GT_ROOT = (
    "./datasets/"  # BECAREFUL, this is the ground truth that comes with GSV-Cities
)


class PittsburghDataset(Dataset):
    def __init__(self, data_dir, which_ds="pitts30k_test", input_transform=None):
        self.data_dir = data_dir

        assert which_ds.lower() in ["pitts30k_val", "pitts30k_test", "pitts250k_test"]

        self.input_transform = input_transform

        # reference images names
        self.dbImages = np.load(GT_ROOT + f"Pittsburgh/{which_ds}_dbImages.npy")

        # query images names
        self.qImages = np.load(GT_ROOT + f"Pittsburgh/{which_ds}_qImages.npy")

        # ground truth
        self.ground_truth = np.load(
            GT_ROOT + f"Pittsburgh/{which_ds}_gt.npy", allow_pickle=True
        )

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index):
        img = Image.open(self.data_dir + "/" + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
