"""
Comprehensive Evaluation Metrics Module
Features: AUC-ROC, PR-AUC, F1, Calibration, ECE
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)
from typing import Dict, Tuple, List, Optional
import warnings


class MetricsCalculator:
    """Comprehensive metrics calculator for binary classification"""
    
    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.y_proba = []
        
    def reset(self):
        """Reset stored predictions"""
        self.y_true = []
        self.y_pred = []
        self.y_proba = []
        
    def update(self, y_true: np.ndarray, y_pred: np.ndarray, 
               y_proba: Optional[np.ndarray] = None):
        """Add predictions to the calculator"""
        self.y_true.extend(y_true.flatten().tolist())
        self.y_pred.extend(y_pred.flatten().tolist())
        if y_proba is not None:
            if len(y_proba.shape) > 1:
                y_proba = y_proba[:, 1]  # Get probability of positive class
            self.y_proba.extend(y_proba.flatten().tolist())
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all metrics"""
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        y_proba = np.array(self.y_proba) if self.y_proba else None
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Probability-based metrics
        if y_proba is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics['auc_roc'] = 0.5
            
            try:
                metrics['pr_auc'] = average_precision_score(y_true, y_proba)
            except ValueError:
                metrics['pr_auc'] = 0.5
            
            # Calibration metrics
            metrics['ece'] = self._expected_calibration_error(y_true, y_proba)
            metrics['mce'] = self._maximum_calibration_error(y_true, y_proba)
        
        return metrics
    
    def _expected_calibration_error(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                     n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        Measures how well predicted probabilities match actual outcomes.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = y_proba[in_bin].mean()
                avg_accuracy = y_true[in_bin].mean()
                ece += prop_in_bin * abs(avg_accuracy - avg_confidence)
        
        return ece
    
    def _maximum_calibration_error(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    n_bins: int = 10) -> float:
        """Compute Maximum Calibration Error (MCE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0
        
        for i in range(n_bins):
            in_bin = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i + 1])
            
            if in_bin.sum() > 0:
                avg_confidence = y_proba[in_bin].mean()
                avg_accuracy = y_true[in_bin].mean()
                mce = max(mce, abs(avg_accuracy - avg_confidence))
        
        return mce
    
    def get_classification_report(self) -> str:
        """Get detailed classification report"""
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        
        return classification_report(
            y_true, y_pred,
            target_names=['Real', 'AI-Generated'],
            digits=4
        )
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(
            np.array(self.y_true),
            np.array(self.y_pred),
            labels=[0, 1]
        )


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Monitors a metric and stops training when no improvement is seen.
    """
    
    def __init__(self, 
                 patience: int = 10, 
                 min_delta: float = 0.001,
                 mode: str = 'max',
                 restore_best: bool = True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' or 'min' (maximize or minimize metric)
            restore_best: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_state = None
        self.early_stop = False
        
    def __call__(self, score: float, model=None, epoch: int = 0) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            model: Model to save state from
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if model is not None:
                self.best_state = model.state_dict().copy()
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if model is not None:
                self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def restore_best_weights(self, model):
        """Restore the best model weights"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print(f"Restored best weights from epoch {self.best_epoch + 1}")


class MovingAverage:
    """Exponential moving average for smoothing metrics"""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = None
        
    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self):
        self.value = None


def compute_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    Compute optimal classification threshold using Youden's J statistic.
    
    Returns:
        (optimal_threshold, best_f1_score)
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1


if __name__ == "__main__":
    # Test metrics
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    y_proba = np.array([0.2, 0.6, 0.8, 0.9, 0.4, 0.3, 0.7, 0.1])
    
    calc = MetricsCalculator()
    calc.update(y_true, y_pred, y_proba)
    
    metrics = calc.compute_all_metrics()
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    print("\n" + calc.get_classification_report())
