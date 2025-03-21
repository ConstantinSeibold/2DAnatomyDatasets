import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.transforms import get_transform
from data import PAXRay166_binary_Dataset
from custom.model import get_model

# Configurations
root_path = os.getenv("PAXRAY_ROOT_PATH", "../../datasets/paxray")
json_path = os.path.join(root_path, "paxray.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TensorBoard Writer
writer = SummaryWriter("runs/training_logs")

# Dataset and DataLoader
train_transform = get_transform("train")
train_dataset = PAXRay166_binary_Dataset(
    root_dir=root_path, 
    splits_json=json_path, 
    split="train", 
    transform=train_transform
)
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)

val_transform = get_transform("val")
val_dataset = PAXRay166_binary_Dataset(
    root_dir=root_path, 
    splits_json=json_path, 
    split="val", 
    transform=val_transform
)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Model Initialization
model = get_model('UNet_resnet50').to(device)

# BCE + Dice Loss
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # Binary Cross-Entropy loss
        bce = self.bce_loss(inputs, targets)

        # Dice coefficient
        smooth = 1e-6
        intersection = (inputs.sigmoid() * targets).sum(dim=(1, 2), keepdim=True)
        union = inputs.sigmoid().sum(dim=(1, 2), keepdim=True) + targets.sum(dim=(1, 2), keepdim=True)
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()  # Dice loss is 1 - mean dice coefficient

        # Combined loss: BCE + Dice loss
        return bce + dice_loss

criterion = BCEDiceLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Learning Rate Scheduler for Decay
def adjust_learning_rate(optimizer, epoch):
    if epoch in [60, 90, 100]:
        lr = optimizer.param_groups[0]['lr'] * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"Learning rate adjusted to {lr}")

# Dice Score Metric Calculation
def dice_score(inputs, targets):
    smooth = 1e-6
    intersection = (inputs.sigmoid() * targets).sum(dim=(2, 3), keepdim=True)
    union = inputs.sigmoid().sum(dim=(2, 3), keepdim=True) + targets.sum(dim=(2, 3), keepdim=True)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# Validation Function
def validate():
    model.eval()
    val_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs, labels = batch[0].to(device), batch[1].to(device).float()
            outputs = model({'data': inputs})["logits"]

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            class_dice = dice_score(outputs, labels).squeeze()
            dice_scores += [class_dice]

    # Calculate the average Dice score across classes
    avg_dice_score = torch.concat(dice_scores).mean(0)
    avg_val_loss = val_loss / len(val_loader)

    return avg_val_loss, avg_dice_score

# Training Loop
def train(num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader)):
            inputs, labels = batch[0].to(device), batch[1].to(device).float()
            
            optimizer.zero_grad()
            outputs = model({'data': inputs})["logits"]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)
        
        avg_loss = running_loss / len(train_loader)
        
        # Adjust learning rate if necessary
        adjust_learning_rate(optimizer, epoch)
        
        # Validate
        avg_val_loss, avg_dice_score = validate()

        # Log metrics
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Mean Dice: {avg_dice_score.mean()}")
        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch+1)
        writer.add_scalar("Loss/val", avg_val_loss, epoch+1)
        for class_idx, dice in enumerate(avg_dice_score):
            writer.add_scalar(f"Dice/class_{class_idx}", dice.item(), epoch+1)

        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
    
    writer.close()

if __name__ == "__main__":
    train(110)
