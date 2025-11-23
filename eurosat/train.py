import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
from typing import Tuple

class EuroSATDataset(Dataset):
    """Dataset class for EuroSAT satellite imagery"""
    
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png', '.tif')):
                    self.samples.append((
                        os.path.join(class_dir, img_name),
                        self.class_to_idx[class_name]
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MultispectralFusion(nn.Module):
    """Custom layer for multispectral feature fusion"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.fusion(x)

class EuroSATClassifier:
    """Classifier for EuroSAT land cover classification"""
    
    def __init__(self, model_name: str = 'resnet50', num_classes: int = 10, 
                 pretrained: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model(model_name, num_classes, pretrained)
        self.model = self.model.to(self.device)
        self.num_classes = num_classes
        
    def _create_model(self, model_name: str, num_classes: int, 
                     pretrained: bool) -> nn.Module:
        """Create model based on architecture name"""
        
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        elif model_name == 'efficientnet_b3':
            model = models.efficientnet_b3(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 
                                           num_classes)
            
        elif model_name == 'vit':
            model = models.vit_b_16(pretrained=pretrained)
            model.heads[0] = nn.Linear(model.heads[0].in_features, num_classes)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model
    
    def train_epoch(self, dataloader: DataLoader, criterion, optimizer) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, dataloader: DataLoader, criterion) -> Tuple[float, float]:
        """Evaluate model on validation/test set"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
             num_epochs: int = 50, lr: float = 0.001):
        """Full training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                         patience=5, factor=0.5)
        
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            print('-' * 60)
            
            scheduler.step(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f'Saved best model with accuracy: {best_acc:.2f}%')
        
        print(f'\nBest validation accuracy: {best_acc:.2f}%')
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

def get_transforms(augment: bool = True):
    """Get data transforms for training/validation"""
    
    if augment:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

def main():
    """Main training script"""
    
    # Configuration
    TRAIN_DIR = 'data/EuroSAT/train'
    VAL_DIR = 'data/EuroSAT/val'
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    MODEL_NAME = 'resnet50'  # Options: resnet50, efficientnet_b3, vit
    NUM_CLASSES = 10
    
    # Create datasets
    train_dataset = EuroSATDataset(TRAIN_DIR, transform=get_transforms(augment=True))
    val_dataset = EuroSATDataset(VAL_DIR, transform=get_transforms(augment=False))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                          shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create classifier and train
    classifier = EuroSATClassifier(model_name=MODEL_NAME, 
                                  num_classes=NUM_CLASSES, 
                                  pretrained=True)
    
    print(f"\nTraining {MODEL_NAME} on {classifier.device}")
    print(f"Total parameters: {sum(p.numel() for p in classifier.model.parameters())}")
    
    classifier.train(train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE)
    
    # Save final model
    classifier.save_model('final_model.pth')

if __name__ == "__main__":
    main()
