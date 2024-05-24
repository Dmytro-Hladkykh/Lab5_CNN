import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
import pickle

logger = logging.getLogger(__name__)

label_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

class CustomDataset(Dataset):
    def __init__(self, data_dir, label_file):
        self.data_dir = data_dir
        self.label_file = label_file
        self.image_files, self.labels = self.load_data()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_data(self):
        image_files = []
        labels = {}
        with open(self.label_file, "r") as f:
            for line in f:
                filename, label = line.strip().split(",")
                image_path = os.path.join(self.data_dir, "images", filename)
                if os.path.exists(image_path):
                    image_files.append(image_path)
                    labels[filename] = label
                else:
                    logger.warning(f"Image file not found: {image_path}")
        return image_files, labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        logger.debug(f"Attempting to load image: {img_path}")
        try:
            image = Image.open(img_path)
            logger.debug(f"Successfully loaded image: {img_path}")
        except Exception as e:
            logger.warning(f"Error reading image {img_path}: {e}")
            return None, None  
        
        image = self.transform(image)
        filename = os.path.basename(img_path)
        label = self.labels[filename]
        label = label_to_idx[label]  
        return image, label

def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for batch_id in config.selected_batches:
        batch_dir = os.path.join("data", "batches", f"batch_{batch_id}")
        label_file = os.path.join(batch_dir, "batch_registry_table.txt")
        train_dataset = CustomDataset(data_dir=batch_dir, label_file=label_file)
        val_dataset = CustomDataset(data_dir=batch_dir, label_file=label_file)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    test_label_file = os.path.join(config.test_batch_dir, "batch_registry_table.txt")
    test_dataset = CustomDataset(data_dir=config.test_batch_dir, label_file=test_label_file)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def collate_fn(batch):
    images = []
    labels = []
    for image, label in batch:
        if image is not None and label is not None:
            if label in label_to_idx:
                images.append(image)
                labels.append(label_to_idx[label])
            else:
                logger.warning(f"Skipping image with unknown label: {label}")

    if not images:
        return None, None  

    images = torch.stack(images)

    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels