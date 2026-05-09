import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.models import load_model
import os

def plot_prediction_sample(model, X_test, y_test, model_name, n_samples=50):
    os.makedirs('results', exist_ok=True)
    y_pred = (model.predict(X_test[:n_samples]) > 0.5).astype(int).flatten()
    y_actual = y_test[:n_samples]
    
    plt.figure(figsize=(14, 4))
    plt.plot(y_actual, label='Actual', marker='o', markersize=3, linewidth=1)
    plt.plot(y_pred, label='Predicted', marker='x', markersize=3, linewidth=1, linestyle='--')
    plt.title(f'{model_name} — Actual vs Predicted Market Direction')
    plt.xlabel('Time Step')
    plt.ylabel('Direction (0=Down, 1=Up)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/predictions_{model_name.lower()}.png')
    plt.show()

if __name__ == "__main__":
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    for model_name in ['RNN', 'LSTM', 'GRU']:
        try:
            model = load_model(f'models/best_{model_name.lower()}.keras')
            plot_prediction_sample(model, X_test, y_test, model_name)
        except Exception as e:
            print(f"Could not load {model_name}: {e}")
    
    print("Visualization complete! Check the results/ folder")