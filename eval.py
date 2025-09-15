import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import os

from vpr_model import VPRModel
from utils.validation import get_validation_recalls

# Dataloader
from dataloaders.val.NordlandDataset import NordlandDataset
from dataloaders.val.MapillaryDataset import MSLS
from dataloaders.val.MapillaryTestDataset import MSLSTest
from dataloaders.val.PittsburghDataset import PittsburghDataset
from dataloaders.val.SPEDDataset import SPEDDataset


def input_transform(image_size=None):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    if image_size:
        return T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        return T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


def get_val_dataset(dataset_name, data_dir="./", image_size=None):
    dataset_name = dataset_name.lower()
    transform = input_transform(image_size=image_size)

    if "nordland" in dataset_name:
        ds = NordlandDataset(data_dir=data_dir + "/Nordland", input_transform=transform)

    elif "msls_test" in dataset_name:
        ds = MSLSTest(data_dir=data_dir + "/msls_test", input_transform=transform)

    elif "msls" in dataset_name:
        ds = MSLS(data_dir=data_dir + "/msls", input_transform=transform)

    elif "pitts" in dataset_name:
        ds = PittsburghDataset(
            data_dir=data_dir + "/Pittsburgh250k", which_ds=dataset_name, input_transform=transform
        )

    elif "sped" in dataset_name:
        ds = SPEDDataset(data_dir=data_dir + "/SPEDTEST", input_transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    num_references = ds.num_references
    num_queries = ds.num_queries
    ground_truth = ds.ground_truth
    return ds, num_references, num_queries, ground_truth


def get_descriptors(model, dataloader, device):
    descriptors = []
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for batch in tqdm(dataloader, "Calculating descritptors..."):
                imgs, labels = batch
                output = model(imgs.to(device)).cpu()
                descriptors.append(output)

    return torch.cat(descriptors)


def load_model(ckpt_path, model_config):
    model = VPRModel(**model_config)
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    model = model.eval()
    model = model.to("cuda")
    print(f"Loaded model from {ckpt_path} Successfully!")
    return model


@hydra.main(version_base="1.3", config_path="configs", config_name="eval")
def main(cfg: DictConfig):
    torch.backends.cudnn.benchmark = True

    # Print config
    print("Evaluation configuration:")
    print(f"Checkpoint: {cfg.ckpt_path}")
    print(f"Datasets: {cfg.val_datasets}")
    print(f"Image size: {cfg.image_size}")
    print(f"Batch size: {cfg.batch_size}")

    model = load_model(cfg.ckpt_path, cfg.model)

    # Parse image size
    image_size = None
    if cfg.image_size is not None:
        if isinstance(cfg.image_size, (list, tuple)) and len(cfg.image_size) == 2:
            image_size = tuple(cfg.image_size)
        elif isinstance(cfg.image_size, int):
            image_size = (cfg.image_size, cfg.image_size)
        else:
            image_size = tuple(cfg.image_size)

    data_dir = cfg.paths.get("data_dir", "./")

    for val_name in cfg.val_datasets:
        val_dataset, num_references, num_queries, ground_truth = get_val_dataset(
            val_name, data_dir, image_size
        )
        val_loader = DataLoader(
            val_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=cfg.pin_memory,
        )

        print(f"Evaluating on {val_name}")
        descriptors = get_descriptors(model, val_loader, "cuda")

        print(f"Descriptor dimension {descriptors.shape[1]}")
        r_list = descriptors[:num_references]
        q_list = descriptors[num_references:]

        print("total_size", descriptors.shape[0], num_queries + num_references)

        testing = isinstance(val_dataset, MSLSTest)

        preds = get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=cfg.k_values,
            gt=ground_truth,
            print_results=cfg.print_results,
            dataset_name=val_name,
            faiss_gpu=cfg.faiss_gpu,
            testing=testing,
        )

        if testing and cfg.save_predictions:
            output_path = cfg.ckpt_path + "." + model.agg_arch + ".preds.txt"
            val_dataset.save_predictions(preds, output_path)

        del descriptors
        print("========> DONE!\n\n")


if __name__ == "__main__":
    main()
