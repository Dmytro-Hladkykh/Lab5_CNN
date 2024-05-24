import os
import torch
import json
import wandb  
from model import ResNet50
from utils import setup_logger
from data import get_data_loaders

def evaluate_model(config):
    logger = setup_logger()

    os.environ["WANDB_API_KEY"] = config.wandb_api_key
    wandb.init(project="cnn_evaluation_experiment", config=config.__dict__) 

    _, _, test_loader = get_data_loaders(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50().to(device)
    model.load_state_dict(torch.load(config.model_path))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    wandb.log({"test_accuracy": accuracy})  

    with open("metrics.json", "w") as f:
        json.dump({"accuracy": accuracy}, f)
    wandb.save("metrics.json")  

if __name__ == "__main__":
    from config import Config
    config = Config()
    evaluate_model(config)
