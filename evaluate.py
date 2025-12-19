"""
Evaluation script for deepfake detection model
Computes comprehensive metrics and visualizations
"""
import torch
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from pipeline import DeepfakeDetectionPipeline
from config import MODEL_DIR, TEST_DIR


def evaluate_model(model_path: str, test_dir: str, device: str = "cpu"):
    """Evaluate model on test set"""
    
    # Load pipeline
    print(f"Loading model from {model_path}...")
    pipeline = DeepfakeDetectionPipeline(model_path=model_path, device=device)
    
    # Load test data
    test_dir = Path(test_dir)
    real_dir = test_dir / "real"
    fake_dir = test_dir / "fake"
    
    # Collect test images
    real_images = []
    if real_dir.exists():
        real_images = list(real_dir.glob("*.jpg")) + \
                     list(real_dir.glob("*.png")) + \
                     list(real_dir.glob("*.jpeg"))
    
    fake_images = []
    if fake_dir.exists():
        fake_images = list(fake_dir.glob("*.jpg")) + \
                     list(fake_dir.glob("*.png")) + \
                     list(fake_dir.glob("*.jpeg"))
    
    print(f"Found {len(real_images)} real images and {len(fake_images)} fake images")
    
    # Predictions
    y_true = []
    y_pred = []
    y_proba = []
    image_paths = []
    
    print("\nRunning predictions...")
    for img_path in tqdm(real_images, desc="Processing real images"):
        try:
            result = pipeline.predict(str(img_path))
            y_true.append(0)  # Real = 0
            y_pred.append(0 if result['prediction'] == "Real Camera Image" else 1)
            y_proba.append(result['ai_generated_probability'])
            image_paths.append(str(img_path))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    for img_path in tqdm(fake_images, desc="Processing fake images"):
        try:
            result = pipeline.predict(str(img_path))
            y_true.append(1)  # Fake = 1
            y_pred.append(0 if result['prediction'] == "Real Camera Image" else 1)
            y_proba.append(result['ai_generated_probability'])
            image_paths.append(str(img_path))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  Real Images:")
    print(f"    Precision: {precision_per_class[0]:.4f}")
    print(f"    Recall:    {recall_per_class[0]:.4f}")
    print(f"    F1-Score:  {f1_per_class[0]:.4f}")
    print(f"  AI-Generated Images:")
    print(f"    Precision: {precision_per_class[1]:.4f}")
    print(f"    Recall:    {recall_per_class[1]:.4f}")
    print(f"    F1-Score:  {f1_per_class[1]:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Real    Fake")
    print(f"  Real    {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"  Fake    {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                              target_names=['Real', 'AI-Generated']))
    
    # Visualizations
    create_visualizations(cm, fpr, tpr, roc_auc, accuracy, precision, recall, f1)
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'num_samples': len(y_true),
        'num_real': int(np.sum(y_true == 0)),
        'num_fake': int(np.sum(y_true == 1)),
    }
    
    results_path = "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return results


def create_visualizations(cm, fpr, tpr, roc_auc, accuracy, precision, recall, f1):
    """Create visualization plots"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'AI-Generated'],
                yticklabels=['Real', 'AI-Generated'],
                ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # ROC Curve
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc="lower right")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('evaluation_plots.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to evaluation_plots.png")
    plt.close()
    
    # Metrics bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    bars = ax.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.4f}',
               ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('metrics_chart.png', dpi=300, bbox_inches='tight')
    print("Metrics chart saved to metrics_chart.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detection Model")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model file (default: best_model.pth)")
    parser.add_argument("--test_dir", type=str, default=TEST_DIR,
                       help="Path to test directory")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model is None:
        model_path = Path(MODEL_DIR) / "best_model.pth"
    else:
        model_path = args.model
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using train.py")
    else:
        evaluate_model(str(model_path), args.test_dir, args.device)

