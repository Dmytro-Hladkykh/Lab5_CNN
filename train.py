import os
import torch
import wandb
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from config import Config
from model import ResNet50
from utils import setup_logger, plot_training_curves
from train_utils import train_model as train_fn

def get_data_loaders(config):
    transform = transforms.Compose([transforms.ToTensor()])
    train_datasets = []
    val_datasets = []
    
    for batch_num in config.selected_batches:
        batch_dir = os.path.join(config.data_dir, f'batch_{batch_num}')
        dataset = datasets.ImageFolder(root=batch_dir, transform=transform)
        
        train_idx, val_idx = train_test_split(list(range(len(dataset))), train_size=config.train_val_split)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
    
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    test_batch_dir = config.test_batch_dir
    test_dataset = datasets.ImageFolder(root=test_batch_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_model(config):
    logger = setup_logger()

    os.environ["WANDB_API_KEY"] = config.wandb_api_key
    wandb.login(key=config.wandb_api_key)  
    wandb.init(project="cnn_training_experiment", config=config.__dict__)  

    train_loader, val_loader, test_loader = get_data_loaders(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50().to(device)

    with wandb.init():
        wandb.config.update(config)  
        train_losses, val_losses, val_accuracies = train_fn(model, train_loader, val_loader,
                                                            config.num_epochs, config.lr, device, logger)
        
        for epoch, (train_loss, val_loss, val_acc) in enumerate(zip(train_losses, val_losses, val_accuracies)):
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}, step=epoch)

        model_save_path = os.path.join(wandb.run.dir, "trained_model.pth")
        torch.save(model.state_dict(), model_save_path)
        wandb.save(model_save_path)

        wandb.run.summary["model_version"] = wandb.run.id  

        plot_training_curves(train_losses, val_losses, torch.tensor(val_accuracies))

if __name__ == "__main__":
    config = Config()
    train_model(config)
