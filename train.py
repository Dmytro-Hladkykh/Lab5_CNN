import os
import torch
import wandb
from config import Config
from data import get_data_loaders
from model import ResNet50
from utils import setup_logger, plot_training_curves
from train_utils import train_model as train_fn

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

        model_path = os.path.join(os.getcwd(), config.model_path)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved locally to: {model_path}")

        wandb_model_path = os.path.join(wandb.run.dir, "trained_model.pth")
        os.makedirs(os.path.dirname(wandb_model_path), exist_ok=True)
        torch.save(model.state_dict(), wandb_model_path)
        wandb.save(wandb_model_path)
        wandb.run.summary["model_version"] = wandb.run.id

        plot_training_curves(train_losses, val_losses, torch.tensor(val_accuracies))

if __name__ == "__main__":
    config = Config()
    train_model(config)
