import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train import EuroSATClassifier, EuroSATDataset, get_transforms
from torch.utils.data import DataLoader

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, classifier: EuroSATClassifier, dataloader: DataLoader, 
                 class_names: list):
        self.classifier = classifier
        self.dataloader = dataloader
        self.class_names = class_names
        self.device = classifier.device
        
    def get_predictions(self):
        """Get all predictions and ground truth labels"""
        self.classifier.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs = inputs.to(self.device)
                outputs = self.classifier.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                from eurosat import evaluate as eurosat_evaluate


                def main():
                    """Wrapper that calls package evaluation entrypoint."""
                    eurosat_evaluate.main()


                if __name__ == "__main__":
                    main()