import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .train import EuroSATClassifier, EuroSATDataset, get_transforms
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
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def compute_metrics(self):
        """Compute classification metrics"""
        preds, labels, probs = self.get_predictions()
        
        # Overall accuracy
        accuracy = np.mean(preds == labels) * 100
        
        # Per-class metrics
        report = classification_report(labels, preds, 
                                      target_names=self.class_names,
                                      digits=4)
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'predictions': preds,
            'labels': labels,
            'probabilities': probs
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_class_accuracy(self, cm: np.ndarray, save_path: str = None):
        """Plot per-class accuracy bar chart"""
        per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.class_names, per_class_acc)
        plt.xlabel('Class')
        plt.ylabel('Accuracy (%)')
        plt.title('Per-Class Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, 105])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_errors(self, top_k: int = 5):
        """Analyze most common misclassifications"""
        preds, labels, _ = self.get_predictions()
        
        errors = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j:
                    count = np.sum((labels == i) & (preds == j))
                    if count > 0:
                        errors.append({
                            'true': self.class_names[i],
                            'pred': self.class_names[j],
                            'count': count
                        })
        
        errors.sort(key=lambda x: x['count'], reverse=True)
        
        print("\nTop Misclassifications:")
        print("-" * 60)
        for error in errors[:top_k]:
            print(f"{error['true']:20s} -> {error['pred']:20s}: {error['count']:3d}")
    
    def evaluate(self, save_plots: bool = True):
        """Run complete evaluation"""
        print("Evaluating model...")
        metrics = self.compute_metrics()
        
        print(f"\nOverall Accuracy: {metrics['accuracy']:.2f}%")
        print("\nClassification Report:")
        print(metrics['report'])
        
        if save_plots:
            self.plot_confusion_matrix(metrics['confusion_matrix'], 
                                      'confusion_matrix.png')
            self.plot_per_class_accuracy(metrics['confusion_matrix'],
                                        'per_class_accuracy.png')
        
        self.analyze_errors()
        
        return metrics

def compare_models(test_loader: DataLoader, class_names: list):
    """Compare different model architectures"""
    models = ['resnet50', 'efficientnet_b3', 'vit']
    results = {}
    
    for model_name in models:
        print(f"\nEvaluating {model_name}...")
        classifier = EuroSATClassifier(model_name=model_name, 
                                      num_classes=len(class_names))
        
        try:
            classifier.load_model(f'{model_name}_best.pth')
        except:
            print(f"Model weights not found for {model_name}")
            continue
        
        evaluator = ModelEvaluator(classifier, test_loader, class_names)
        metrics = evaluator.compute_metrics()
        results[model_name] = metrics['accuracy']
    
    # Plot comparison
    if results:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(results.keys(), results.values())
        plt.ylabel('Accuracy (%)')
        plt.title('Model Architecture Comparison')
        plt.ylim([0, 105])
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main evaluation script"""
    
    # Configuration
    TEST_DIR = 'data/EuroSAT/test'
    MODEL_PATH = 'best_model.pth'
    MODEL_NAME = 'resnet50'
    BATCH_SIZE = 32
    NUM_CLASSES = 10
    
    # Load test dataset
    test_dataset = EuroSATDataset(TEST_DIR, transform=get_transforms(augment=False))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=4)
    
    class_names = test_dataset.classes
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {class_names}")
    
    # Load trained model
    classifier = EuroSATClassifier(model_name=MODEL_NAME, 
                                  num_classes=NUM_CLASSES,
                                  pretrained=False)
    classifier.load_model(MODEL_PATH)
    
    # Evaluate
    evaluator = ModelEvaluator(classifier, test_loader, class_names)
    metrics = evaluator.evaluate(save_plots=True)
    
    # Optional: Compare multiple models
    # compare_models(test_loader, class_names)

if __name__ == "__main__":
    main()
