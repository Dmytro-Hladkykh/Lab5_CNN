import os
import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import wandb
from model import ResNet50
from data import get_data_loaders
from utils import setup_logger

def interpret_model(config):
    logger = setup_logger()

    os.environ["WANDB_API_KEY"] = config.wandb_api_key
    wandb.login(key=config.wandb_api_key)  
    wandb.init(project="cnn_interpretation_experiment", config=config.__dict__)  

    _, _, test_loader = get_data_loaders(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50().to(device)
    model.load_state_dict(torch.load(config.model_path))
    model.eval()

    ig = IntegratedGradients(model)
    
    output_dir = "interpretation_results"
    os.makedirs(output_dir, exist_ok=True)

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        attributions, delta = ig.attribute(images, target=labels, return_convergence_delta=True)
        for i in range(len(images)):
            attr_img = attributions[i].cpu().detach().numpy().transpose(1, 2, 0)
            plt.imshow(attr_img)
            plt.axis('off')
            img_path = os.path.join(output_dir, f"attr_{i}.png")
            plt.savefig(img_path)
            plt.close()  
            wandb.log({"interpretation": [wandb.Image(img_path, caption=f"Label: {labels[i].item()}")]})
            os.remove(img_path)  
            
if __name__ == "__main__":
    from config import Config
    config = Config()
    interpret_model(config)