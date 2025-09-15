from pathlib import Path
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset

# NOTE: you need to download the Nordland dataset from  https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W
# this link is shared and maintained by the authors of VPR_Bench: https://github.com/MubarizZaffar/VPR-Bench
# the folders named ref and query should reside in DATASET_ROOT path
# I hardcoded the image names and ground truth for faster evaluation
# performance is exactly the same as if you use VPR-Bench.

GT_ROOT = "./datasets/"


class NordlandDataset(Dataset):
    def __init__(self, data_dir, input_transform=None):
        self.data_dir = data_dir

        self.input_transform = input_transform

        # reference images names
        self.dbImages = np.load(GT_ROOT + "Nordland/Nordland_dbImages.npy")

        # query images names
        self.qImages = np.load(GT_ROOT + "Nordland/Nordland_qImages.npy")

        # ground truth
        self.ground_truth = np.load(
            GT_ROOT + "Nordland/Nordland_gt.npy", allow_pickle=True
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
