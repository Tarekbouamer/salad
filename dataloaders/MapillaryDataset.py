from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# NOTE: you need to download the mapillary_sls dataset from  https://github.com/FrederikWarburg/mapillary_sls
# make sure the path where the mapillary_sls validation dataset resides on your computer is correct.
# the folder named train_val should reside in DATASET_ROOT path (that's the only folder you need from mapillary_sls)
# I hardcoded the groundtruth for image to image evaluation, otherwise it would take ages to run the groundtruth script at each epoch.


GT_ROOT = str(Path.cwd() / "datasets")


class MSLS(Dataset):
    def __init__(self, data_dir, input_transform=None):
        self.data_dir = data_dir + "/msls"

        self.input_transform = input_transform

        # hard coded reference image names, this avoids the hassle of listing them at each epoch.
        self.dbImages = np.load(GT_ROOT + "/msls_val/msls_val_dbImages.npy")

        # hard coded query image names.
        self.qImages = np.load(GT_ROOT + "/msls_val/msls_val_qImages.npy")

        # hard coded index of query images
        self.qIdx = np.load(GT_ROOT + "/msls_val/msls_val_qIdx.npy")

        # hard coded groundtruth (correspondence between each query and its matches)
        self.pIdx = np.load(GT_ROOT + "/msls_val/msls_val_pIdx.npy", allow_pickle=True)

        # concatenate reference images then query images so that we can use only one dataloader
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))

        # we need to keeo the number of references so that we can split references-queries
        # when calculating recall@K
        self.num_references = len(self.dbImages)

    def __getitem__(self, index):
        img = Image.open(self.data_dir + "/" + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
