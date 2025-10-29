import sys
import os

# Add custom TensorFlow installation path only for local Windows environment
if os.name == 'nt' and os.path.exists(r'C:\tf'):
    sys.path.insert(0, r'C:\tf')

import numpy as np
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_PATH = "tf-cnn-model.h5"

def load_test_data():
    """Load MNIST test dataset"""
    print("[INFO] Loading MNIST test dataset...")
    (_, _), (x_test, y_test) = mnist.load_data()
    
    # Preprocess the data
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32') / 255.0
    
    print(f"[INFO] Test set size: {len(x_test)} images")
    return x_test, y_test

def evaluate_model():
    """Evaluate the model and calculate metrics"""
    print("[INFO] Loading model...")
    model = models.load_model(MODEL_PATH, compile=False)
    print("[INFO] Model loaded successfully!")
    
    # Load test data
    x_test, y_test = load_test_data()
    
    # Make predictions
    print("[INFO] Making predictions on test set...")
    predictions = model.predict(x_test, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100
    
    print(f"\nTable 1: Model Performance Metrics")
    print("-" * 50)
    print(f"Accuracy:  {accuracy:.3f}%")
    print(f"Precision: {precision:.3f}%")
    print(f"Recall:    {recall:.3f}%")
    print(f"F1 Score:  {f1:.3f}%")
    print("-" * 50)
    
    # Detailed classification report
    print("\n\nDetailed Classification Report:")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))
    
    # Confusion Matrix
    print("\n[INFO] Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix - Handwritten Digit Recognition', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("[INFO] Confusion matrix saved as 'confusion_matrix.png'")
    
    # Per-class accuracy
    print("\n\nPer-Class Accuracy:")
    print("-" * 50)
    for i in range(10):
        class_correct = cm[i, i]
        class_total = cm[i, :].sum()
        class_accuracy = (class_correct / class_total) * 100
        print(f"Digit {i}: {class_accuracy:.2f}% ({class_correct}/{class_total})")
    print("-" * 50)
    
    # Save metrics to file
    with open('model_metrics.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Table 1: Model Performance Metrics\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy:  {accuracy:.3f}%\n")
        f.write(f"Precision: {precision:.3f}%\n")
        f.write(f"Recall:    {recall:.3f}%\n")
        f.write(f"F1 Score:  {f1:.3f}%\n")
        f.write("-" * 50 + "\n\n")
        f.write("\nDetailed Classification Report:\n")
        f.write("="*60 + "\n")
        f.write(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))
        f.write("\n\nPer-Class Accuracy:\n")
        f.write("-" * 50 + "\n")
        for i in range(10):
            class_correct = cm[i, i]
            class_total = cm[i, :].sum()
            class_accuracy = (class_correct / class_total) * 100
            f.write(f"Digit {i}: {class_accuracy:.2f}% ({class_correct}/{class_total})\n")
        f.write("-" * 50 + "\n")
    
    print("\n[INFO] Metrics saved to 'model_metrics.txt'")
    
    # Create metrics visualization
    create_metrics_plot(accuracy, precision, recall, f1)
    
    return accuracy, precision, recall, f1

def create_metrics_plot(accuracy, precision, recall, f1):
    """Create a bar plot of the metrics"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylim(0, 105)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    print("[INFO] Performance metrics plot saved as 'performance_metrics.png'")
    plt.close()

if __name__ == "__main__":
    try:
        print("\n" + "="*60)
        print("MNIST HANDWRITTEN DIGIT RECOGNITION - MODEL EVALUATION")
        print("="*60 + "\n")
        
        accuracy, precision, recall, f1 = evaluate_model()
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("  1. model_metrics.txt - Detailed metrics report")
        print("  2. confusion_matrix.png - Confusion matrix visualization")
        print("  3. performance_metrics.png - Metrics bar chart")
        print("\n")
        
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

