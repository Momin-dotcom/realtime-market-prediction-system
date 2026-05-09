import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, 
                             confusion_matrix, classification_report)
from tensorflow.keras.models import load_model
import os

def evaluate_model(model, X_test, y_test, model_name):
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    return acc, f1, cm, y_pred

def plot_confusion_matrix(cm, model_name):
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'])
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{model_name.lower()}.png')
    plt.show()
    print(f"Saved confusion matrix for {model_name}")

def plot_training_history(history, model_name):
    os.makedirs('results', exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title(f'{model_name} — Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title(f'{model_name} — Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/training_history_{model_name.lower()}.png')
    plt.show()
    print(f"Saved training history for {model_name}")

def compare_models(results):
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    f1_scores = [results[m]['f1'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='coral')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — Accuracy and F1 Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    plt.show()
    print("Saved model comparison chart")

if __name__ == "__main__":
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    results = {}
    for model_name in ['RNN', 'LSTM', 'GRU']:
        model = load_model(f'models/best_{model_name.lower()}.keras')
        acc, f1, cm, y_pred = evaluate_model(model, X_test, y_test, model_name)
        plot_confusion_matrix(cm, model_name)
        results[model_name] = {'accuracy': acc, 'f1': f1}
    
    compare_models(results)
    print("\nAll evaluations complete!")