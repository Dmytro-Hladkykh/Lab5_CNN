# train_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import setup_logger, plot_training_curves

logger = setup_logger()

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    skipped_batches = 0
    for i, (images, labels) in enumerate(data_loader):
        if images is None or labels is None:
            logger.warning(f"Skipping empty batch {i}: images or labels are None")
            skipped_batches += 1
            continue  
        logger.debug(f"Processing batch {i}")
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    logger.info(f"Skipped {skipped_batches} batches during training")
    return running_loss / total_samples if total_samples > 0 else 0

def validate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    skipped_batches = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            if images is None or labels is None:
                logger.warning(f"Skipping empty batch {i}: images or labels are None")
                skipped_batches += 1
                continue  
            logger.debug(f"Processing batch {i}")
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
    logger.info(f"Skipped {skipped_batches} batches during validation")
    val_loss = running_loss / total_samples if total_samples > 0 else 0
    val_acc = running_corrects / total_samples if total_samples > 0 else 0
    return val_loss, val_acc

def early_stopping(val_loss, best_val_loss, patience, counter):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
    return best_val_loss, counter

def train_model(model, train_loader, val_loader, num_epochs, lr, device, logger, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        best_val_loss, early_stop_counter = early_stopping(val_loss, best_val_loss, patience, early_stop_counter)

        if early_stop_counter >= patience:
            logger.info("Early stopping triggered. Training stopped.")
            break

    return train_losses, val_losses, val_accuracies
